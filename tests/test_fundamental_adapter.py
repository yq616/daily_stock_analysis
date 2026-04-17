# -*- coding: utf-8 -*-
"""
Tests for fundamental adapter helpers.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_provider.fundamental_adapter import (
    AkshareFundamentalAdapter,
    _build_financial_abstract_payload,
    _build_dividend_payload,
    _extract_latest_row,
    _parse_dividend_plan_to_per_share,
    _recent_report_dates,
    _report_date_to_institute_period,
    _with_akshare_market_prefix,
)


class TestFundamentalAdapter(unittest.TestCase):
    def test_parse_dividend_plan_to_per_share_supports_cn_patterns(self) -> None:
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("10派3元(含税)"), 0.3, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每10股派发2.5元"), 0.25, places=6)
        self.assertAlmostEqual(_parse_dividend_plan_to_per_share("每股派0.8元"), 0.8, places=6)
        self.assertIsNone(_parse_dividend_plan_to_per_share("仅送股，不现金分红"))

    def test_extract_latest_row_returns_none_when_code_mismatch(self) -> None:
        df = pd.DataFrame(
            {
                "股票代码": ["600000", "000001"],
                "值": [1, 2],
            }
        )
        row = _extract_latest_row(df, "600519")
        self.assertIsNone(row)

    def test_extract_latest_row_fallback_when_no_code_column(self) -> None:
        df = pd.DataFrame({"值": [1, 2]})
        row = _extract_latest_row(df, "600519")
        self.assertIsNotNone(row)
        self.assertEqual(row["值"], 1)

    def test_market_prefix_helper_supports_a_share_codes(self) -> None:
        self.assertEqual(_with_akshare_market_prefix("603269"), "sh603269")
        self.assertEqual(_with_akshare_market_prefix("300750"), "sz300750")
        self.assertEqual(_with_akshare_market_prefix("830799"), "bj830799")
        self.assertEqual(_with_akshare_market_prefix("sh600519"), "sh600519")

    def test_report_date_helpers_generate_expected_formats(self) -> None:
        dates = _recent_report_dates(now=datetime(2026, 4, 12), limit=4)
        self.assertEqual(dates, ["20260331", "20251231", "20250930", "20250630"])
        self.assertEqual(_report_date_to_institute_period("20250331"), "20251")
        self.assertEqual(_report_date_to_institute_period("20250630"), "20252")
        self.assertEqual(_report_date_to_institute_period("20250930"), "20253")
        self.assertEqual(_report_date_to_institute_period("20251231"), "20254")
        self.assertIsNone(_report_date_to_institute_period("20250115"))

    def test_build_financial_abstract_payload_supports_wide_table(self) -> None:
        df = pd.DataFrame(
            {
                "选项": ["成长能力", "成长能力", "常用指标", "常用指标", "常用指标", "盈利能力", "盈利能力"],
                "指标": [
                    "营业总收入增长率",
                    "归属母公司净利润增长率",
                    "营业总收入",
                    "归母净利润",
                    "经营现金流量净额",
                    "净资产收益率(ROE)",
                    "毛利率",
                ],
                "20260331": [8.2, 9.4, 120.0, 24.0, 18.0, 10.5, 32.1],
                "20251231": [7.1, 8.3, 400.0, 80.0, 62.0, 9.9, 31.2],
            }
        )

        payload = _build_financial_abstract_payload(df)

        self.assertEqual(payload["financial_report"]["report_date"], "2026-03-31")
        self.assertEqual(payload["financial_report"]["revenue"], 120.0)
        self.assertEqual(payload["financial_report"]["net_profit_parent"], 24.0)
        self.assertEqual(payload["financial_report"]["operating_cash_flow"], 18.0)
        self.assertEqual(payload["growth"]["revenue_yoy"], 8.2)
        self.assertEqual(payload["growth"]["net_profit_yoy"], 9.4)
        self.assertEqual(payload["growth"]["roe"], 10.5)
        self.assertEqual(payload["growth"]["gross_margin"], 32.1)

    def test_fundamental_bundle_falls_back_to_shareholder_count_detail(self) -> None:
        adapter = AkshareFundamentalAdapter()
        fin_df = pd.DataFrame(
            {
                "选项": ["成长能力", "成长能力", "常用指标", "常用指标", "常用指标", "盈利能力", "盈利能力"],
                "指标": [
                    "营业总收入增长率",
                    "归属母公司净利润增长率",
                    "营业总收入",
                    "归母净利润",
                    "经营现金流量净额",
                    "净资产收益率(ROE)",
                    "毛利率",
                ],
                "20260331": [8.2, 9.4, 120.0, 24.0, 18.0, 10.5, 32.1],
            }
        )
        shareholder_df = pd.DataFrame(
            {
                "股东户数统计截止日": ["2026-03-31"],
                "股东户数-本次": [10000],
                "股东户数-上次": [10500],
                "股东户数-增减": [-500],
                "股东户数-增减比例": [-4.7619],
                "户均持股数量": [1234.5],
                "代码": ["603269"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (pd.DataFrame(), None, []),
                (pd.DataFrame(), None, []),
                (pd.DataFrame(), None, []),
                (None, None, []),
                (shareholder_df, "stock_zh_a_gdhs_detail_em", ["stock_gdfx_top_10_em:ValueError"]),
                (shareholder_df, "stock_zh_a_gdhs_detail_em", []),
            ],
        ):
            result = adapter.get_fundamental_bundle("603269")

        self.assertEqual(result["institution"].get("shareholder_count"), 10000.0)
        self.assertEqual(result["institution"].get("shareholder_count_change"), -500.0)
        self.assertEqual(result["institution"].get("shareholder_count_change_pct"), -4.7619)
        self.assertEqual(result["institution"].get("avg_hold_quantity"), 1234.5)
        self.assertEqual(result["institution"].get("holder_snapshot_date"), "2026-03-31")
        self.assertIn("shareholder_count:stock_zh_a_gdhs_detail_em", result["source_chain"])

    def test_dragon_tiger_no_match_with_code_column_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        df = pd.DataFrame(
            {
                "股票代码": ["600000"],
                "日期": ["2026-01-01"],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["is_on_list"])
        self.assertEqual(result["recent_count"], 0)

    def test_dragon_tiger_match_is_ok(self) -> None:
        adapter = AkshareFundamentalAdapter()
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "日期": [today],
            }
        )
        with patch.object(adapter, "_call_df_candidates", return_value=(df, "stock_lhb_stock_statistic_em", [])):
            result = adapter.get_dragon_tiger_flag("600519")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["is_on_list"])
        self.assertGreaterEqual(result["recent_count"], 1)

    def test_fundamental_bundle_includes_financial_report_and_dividend_payload(self) -> None:
        adapter = AkshareFundamentalAdapter()
        now = datetime.now()
        within_ttm = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        future_day = (now + timedelta(days=10)).strftime("%Y-%m-%d")
        old_day = (now - timedelta(days=500)).strftime("%Y-%m-%d")
        fin_df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "报告期": [within_ttm],
                "营业总收入": [1000.0],
                "归母净利润": [300.0],
                "经营活动产生的现金流量净额": [500.0],
                "净资产收益率": [18.2],
                "营业收入同比": [12.0],
                "净利润同比": [9.5],
            }
        )
        forecast_df = pd.DataFrame({"股票代码": ["600519"], "预告": ["预增"]})
        quick_df = pd.DataFrame({"股票代码": ["600519"], "快报": ["快报摘要"]})
        dividend_df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519", "600519", "600519"],
                "除息日": [within_ttm, within_ttm, future_day, old_day],
                "分配方案": ["10派3元(含税)", "10派3元(含税)", "10派5元", "10派1元"],
            }
        )

        with patch.object(
            adapter,
            "_call_df_candidates",
            side_effect=[
                (fin_df, "stock_financial_abstract", []),
                (forecast_df, "stock_yjyg_em", []),
                (quick_df, "stock_yjkb_em", []),
                (dividend_df, "stock_fhps_detail_em", []),
                (None, None, []),
                (None, None, []),
                (None, None, []),
            ],
        ):
            result = adapter.get_fundamental_bundle("600519")

        financial_report = result["earnings"].get("financial_report", {})
        self.assertEqual(financial_report.get("report_date"), within_ttm)
        self.assertEqual(financial_report.get("revenue"), 1000.0)
        self.assertEqual(financial_report.get("net_profit_parent"), 300.0)
        self.assertEqual(financial_report.get("operating_cash_flow"), 500.0)
        self.assertEqual(financial_report.get("roe"), 18.2)

        dividend_payload = result["earnings"].get("dividend", {})
        events = dividend_payload.get("events", [])
        self.assertEqual(len(events), 2)  # duplicate + future day filtered
        self.assertEqual(dividend_payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(dividend_payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_build_dividend_payload_returns_empty_when_code_not_matched(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["000001"],
                "除息日": [now],
                "分配方案": ["10派3元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_skips_after_tax_plan(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519"],
                "除息日": [now],
                "分配方案": ["10派3元(税后)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload, {})

    def test_build_dividend_payload_ttm_window_boundary(self) -> None:
        now = datetime.now()
        day_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
        day_366 = (now - timedelta(days=366)).strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "股票代码": ["600519", "600519"],
                "除息日": [day_365, day_366],
                "分配方案": ["10派3元(含税)", "10派5元(含税)"],
            }
        )

        payload = _build_dividend_payload(df, stock_code="600519")
        self.assertEqual(payload.get("ttm_event_count"), 1)
        self.assertAlmostEqual(payload.get("ttm_cash_dividend_per_share"), 0.3, places=6)

    def test_fundamental_bundle_supports_current_akshare_signatures(self) -> None:
        adapter = AkshareFundamentalAdapter()
        captured_calls = {
            "stock_yjyg_em": [],
            "stock_yjkb_em": [],
            "stock_yjbb_em": [],
            "stock_institute_hold": [],
            "stock_gdfx_top_10_em": [],
        }
        fin_df = pd.DataFrame(
            {
                "选项": ["成长能力", "成长能力", "常用指标", "常用指标", "常用指标", "盈利能力", "盈利能力"],
                "指标": [
                    "营业总收入增长率",
                    "归属母公司净利润增长率",
                    "营业总收入",
                    "归母净利润",
                    "经营现金流量净额",
                    "净资产收益率(ROE)",
                    "毛利率",
                ],
                "20260331": [8.2, 9.4, 120.0, 24.0, 18.0, 10.5, 32.1],
                "20251231": [7.1, 8.3, 400.0, 80.0, 62.0, 9.9, 31.2],
            }
        )
        forecast_market_df = pd.DataFrame({"股票代码": ["603269"], "业绩变动": ["预增"]})
        quick_market_df = pd.DataFrame({"股票代码": ["603269"], "快报摘要": ["净利润稳步增长"]})
        inst_market_df = pd.DataFrame({"证券代码": ["603269"], "机构数变化": [3]})
        top10_df = pd.DataFrame({"股东名称": ["示例股东"], "增减": [5], "变动比率": [1.2]})

        class FakeAkshare:
            @staticmethod
            def stock_financial_abstract(symbol: str) -> pd.DataFrame:
                return fin_df

            @staticmethod
            def stock_yjyg_em(date: str = "20200331") -> pd.DataFrame:
                captured_calls["stock_yjyg_em"].append(date)
                if date == "20260331":
                    return forecast_market_df
                return pd.DataFrame()

            @staticmethod
            def stock_yjbb_em(date: str = "20200331") -> pd.DataFrame:
                captured_calls["stock_yjbb_em"].append(date)
                return pd.DataFrame()

            @staticmethod
            def stock_yjkb_em(date: str = "20211231") -> pd.DataFrame:
                captured_calls["stock_yjkb_em"].append(date)
                if date == "20260331":
                    return quick_market_df
                return pd.DataFrame()

            @staticmethod
            def stock_fhps_detail_em(symbol: str) -> pd.DataFrame:
                return pd.DataFrame()

            @staticmethod
            def stock_institute_hold(symbol: str = "20051") -> pd.DataFrame:
                captured_calls["stock_institute_hold"].append(symbol)
                if symbol == "20261":
                    return inst_market_df
                return pd.DataFrame()

            @staticmethod
            def stock_institute_recommend() -> pd.DataFrame:
                return pd.DataFrame()

            @staticmethod
            def stock_gdfx_top_10_em(symbol: str = "sh688686", date: str = "20210630") -> pd.DataFrame:
                captured_calls["stock_gdfx_top_10_em"].append((symbol, date))
                if symbol == "sh603269" and date == "20260331":
                    return top10_df
                return pd.DataFrame()

            @staticmethod
            def stock_zh_a_gdhs_detail_em(symbol: str) -> pd.DataFrame:
                return pd.DataFrame()

        with patch.dict(sys.modules, {"akshare": FakeAkshare}):
            with patch.object(adapter, "_report_date_candidates", return_value=["20260331", "20251231"]):
                result = adapter.get_fundamental_bundle("603269")

        self.assertEqual(captured_calls["stock_yjyg_em"], ["20260331"])
        self.assertEqual(captured_calls["stock_yjkb_em"], ["20260331"])
        self.assertEqual(captured_calls["stock_institute_hold"], ["20261"])
        self.assertEqual(captured_calls["stock_gdfx_top_10_em"], [("sh603269", "20260331")])
        self.assertEqual(result["earnings"].get("forecast_summary"), "预增")
        self.assertEqual(result["earnings"].get("quick_report_summary"), "净利润稳步增长")
        self.assertEqual(result["institution"].get("institution_holding_change"), 3.0)
        self.assertEqual(result["institution"].get("top10_holder_change"), 5.0)
        self.assertEqual(result["growth"].get("revenue_yoy"), 8.2)
        self.assertEqual(result["growth"].get("net_profit_yoy"), 9.4)
        self.assertEqual(result["status"], "partial")


if __name__ == "__main__":
    unittest.main()
