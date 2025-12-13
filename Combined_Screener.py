# -*- coding: utf-8 -*-
import tushare as ts
import pandas as pd
import datetime
import time
import numpy as np
from datetime import timedelta, datetime as dt
import talib
from tqdm import tqdm
import requests
import json
import os
import traceback

# ==============================================================================
#                                配置区域
# ==============================================================================

# 1. 优先从环境变量获取Token和Webhook (适配GitHub Actions)
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "b249a314d4a8db3e43f44db9d5524f31f3425fde397fc9c4633bf9a9")
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "") 

# 初始化 Tushare
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# API调用之间的延时
API_CALL_DELAY = 0.05 

# --- 飞书发送函数 (新增) ---
def send_feishu_msg(title, content):
    """发送飞书消息的通用函数"""
    if not FEISHU_WEBHOOK_URL:
        # 本地运行时如果没有配置环境变量，只打印不发送，防止报错
        print(f"【模拟发送飞书】标题: {title}")
        return
    
    if not content:
        content = "今日无符合条件的标的。"
    
    # 构造消息
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    full_text = f"【{title}】\n{current_time}\n--------------------\n{content}"
    
    headers = {'Content-Type': 'application/json'}
    payload = {"msg_type": "text", "content": {"text": full_text}}
    
    try:
        resp = requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        if resp.status_code == 200:
            print(f">> 飞书通知已发送: {title}")
        else:
            print(f">> 飞书发送失败: {resp.text}")
    except Exception as e:
        print(f">> 飞书发送报错: {e}")

# ==============================================================================
#                                缓存与辅助函数 (完全保持原样)
# ==============================================================================
historical_data_cache = {}
financial_data_cache = {}
concept_detail_cache = {}
daily_basic_cache = {}
stk_holder_cache = {}
top10_holder_cache = {}
daily_data_cache = {}

def get_last_trade_day(today_pd_timestamp):
    days_back = 0
    _last_trade_day = None
    print("正在查找最近的交易日...")
    while True:
        current_check_date = (today_pd_timestamp - pd.Timedelta(days=days_back))
        last_trade_day_str = current_check_date.strftime('%Y%m%d')
        time.sleep(API_CALL_DELAY)
        trade_cal = pro.trade_cal(exchange='', start_date=last_trade_day_str, end_date=last_trade_day_str)
        if not trade_cal.empty and trade_cal.iloc[0]['is_open'] == 1:
            time.sleep(API_CALL_DELAY)
            daily_data_check = pro.daily(trade_date=last_trade_day_str, limit=1)
            if not daily_data_check.empty:
                _last_trade_day = last_trade_day_str
                print(f"查找到的最近交易日: {_last_trade_day}")
                return _last_trade_day
        days_back += 1
        if days_back > 60:
            print("错误：在过去60天内未能找到有效的交易日。程序退出。")
            exit()

def get_stock_data_ma_cached(ts_code, start_date, end_date, ma_params=(20, 60, 120)):
    cache_key = (ts_code, start_date, end_date, tuple(sorted(ma_params)))
    if cache_key in historical_data_cache:
        return historical_data_cache[cache_key].copy()
    time.sleep(API_CALL_DELAY)
    df = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date, ma=list(ma_params))
    if df is None or df.empty:
        historical_data_cache[cache_key] = None
        return None
    df.sort_values(by='trade_date', ascending=True, inplace=True)
    historical_data_cache[cache_key] = df
    return df.copy()

def get_stock_data_talib_ma_cached(ts_code, start_date, end_date, ma_params=(20, 60, 120)):
    cache_key = (ts_code, start_date, end_date, 'talib_base')
    if cache_key in historical_data_cache:
        df_base = historical_data_cache[cache_key]
        if df_base is None: return None
    else:
        time.sleep(API_CALL_DELAY)
        df_base = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=start_date, end_date=end_date)
        if df_base is None or df_base.empty:
            historical_data_cache[cache_key] = None
            return None
        df_base.sort_values(by='trade_date', ascending=True, inplace=True)
        historical_data_cache[cache_key] = df_base

    df_ma = df_base.copy()
    close_prices = df_ma['close'].values
    for period in ma_params:
        if len(close_prices) >= period:
            df_ma[f'ma{period}'] = talib.SMA(close_prices, timeperiod=period)
        else:
            df_ma[f'ma{period}'] = np.nan
    return df_ma

def get_recent_financials_cached(ts_code):
    if ts_code in financial_data_cache:
        return financial_data_cache[ts_code]
    preferred_report_types = ['1', '2', '6', '7']
    try:
        time.sleep(API_CALL_DELAY)
        income_data = pro.income(ts_code=ts_code, fields='ts_code,ann_date,end_date,report_type,revenue')
        if income_data.empty:
            financial_data_cache[ts_code] = None
            return None
        time.sleep(API_CALL_DELAY)
        fina_indicator_data = pro.fina_indicator(ts_code=ts_code, fields='ts_code,ann_date,end_date,profit_dedt')
        if fina_indicator_data.empty:
            financial_data_cache[ts_code] = None
            return None
        financials = pd.merge(income_data, fina_indicator_data, on=['ts_code', 'ann_date', 'end_date'], how='inner')
        if financials.empty:
            financial_data_cache[ts_code] = None
            return None
        financials['report_type_priority'] = financials['report_type'].astype(str).apply(lambda x: 1 if x in preferred_report_types else 2)
        financials = financials.sort_values(by=['end_date', 'report_type_priority', 'ann_date'], ascending=[False, True, False])
        latest_financials = financials.iloc[0].copy()
        revenue = latest_financials.get('revenue')
        profit_dedt = latest_financials.get('profit_dedt')
        if pd.notna(revenue) and revenue != 0 and pd.notna(profit_dedt):
            latest_financials['deducted_net_profit_margin'] = profit_dedt / revenue
        else:
            latest_financials['deducted_net_profit_margin'] = np.nan
        financial_data_cache[ts_code] = latest_financials
        return latest_financials
    except Exception as e:
        financial_data_cache[ts_code] = None
        return None

def get_concept_detail_cached(ts_code):
    if ts_code in concept_detail_cache:
        return concept_detail_cache[ts_code]
    try:
        time.sleep(API_CALL_DELAY)
        concept_info = pro.concept_detail(ts_code=ts_code)
        concept_detail_cache[ts_code] = concept_info
        return concept_info
    except Exception:
        concept_detail_cache[ts_code] = pd.DataFrame()
        return pd.DataFrame()

def get_daily_basic_cached(ts_code, trade_date):
    cache_key = (ts_code, trade_date)
    if cache_key in daily_basic_cache:
        return daily_basic_cache[cache_key]
    try:
        time.sleep(API_CALL_DELAY)
        daily_basic_info = pro.daily_basic(ts_code=ts_code, trade_date=trade_date)
        daily_basic_cache[cache_key] = daily_basic_info
        return daily_basic_info
    except Exception:
        daily_basic_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

def get_stk_holdernumber_cached(ts_code, start_date, end_date):
    cache_key = (ts_code, start_date, end_date, 'stk_holder')
    if cache_key in stk_holder_cache:
        return stk_holder_cache[cache_key]
    try:
        time.sleep(API_CALL_DELAY)
        holder_info = pro.stk_holdernumber(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='ts_code,end_date,holder_num')
        stk_holder_cache[cache_key] = holder_info
        return holder_info
    except Exception:
        stk_holder_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

def get_top10_floatholders_cached(ts_code, start_date, end_date):
    cache_key = (ts_code, start_date, end_date, 'top10_float')
    if cache_key in top10_holder_cache:
        return top10_holder_cache[cache_key]
    try:
        time.sleep(API_CALL_DELAY)
        top10_info = pro.top10_floatholders(ts_code=ts_code, period='', start_date=start_date, end_date=end_date, fields='ts_code,ann_date,end_date,hold_amount')
        top10_holder_cache[cache_key] = top10_info
        return top10_info
    except Exception:
        top10_holder_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

def get_daily_data_for_period_cached(ts_code, start_date, end_date):
    cache_key = (ts_code, start_date, end_date, 'daily_data')
    if cache_key in daily_data_cache:
        return daily_data_cache[cache_key]
    try:
        time.sleep(API_CALL_DELAY)
        daily_df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        daily_data_cache[cache_key] = daily_df
        return daily_df
    except Exception:
        daily_data_cache[cache_key] = pd.DataFrame()
        return pd.DataFrame()

def classify_growth_detail_s1(ts_code_param):
    """独立获取财务数据并分类增长情况（用于脚本1的步骤7.4）"""
    try:
        preferred_report_types = ['1', '2', '6', '7']
        time.sleep(API_CALL_DELAY); income_data_g = pro.income(ts_code=ts_code_param, fields='ts_code,ann_date,end_date,report_type,revenue')
        if income_data_g.empty: return '数据不足(营收)'

        time.sleep(API_CALL_DELAY); fina_indicator_data_g = pro.fina_indicator(ts_code=ts_code_param, fields='ts_code,ann_date,end_date,profit_dedt')
        if fina_indicator_data_g.empty: return '数据不足(利润)'

        financials_g = pd.merge(income_data_g, fina_indicator_data_g, on=['ts_code', 'ann_date', 'end_date'], how='inner')
        if financials_g.empty: return '数据不足(合并)'

        financials_g['report_type_priority'] = financials_g['report_type'].astype(str).apply(lambda x: 1 if x in preferred_report_types else 2)
        financials_g = financials_g.sort_values(by=['end_date', 'report_type_priority', 'ann_date'], ascending=[False, True, False])
        financials_g = financials_g.drop_duplicates(subset=['end_date'])
        financials_g = financials_g.head(8)
        if len(financials_g) < 8: return '数据不足(少于8期)'

        latest_4_revenue_sum = financials_g.iloc[:4]['revenue'].sum(skipna=True)
        previous_4_revenue_sum = financials_g.iloc[4:8]['revenue'].sum(skipna=True)
        latest_4_profit_sum = financials_g.iloc[:4]['profit_dedt'].sum(skipna=True)
        previous_4_profit_sum = financials_g.iloc[4:8]['profit_dedt'].sum(skipna=True)

        revenue_increased = (latest_4_revenue_sum > previous_4_revenue_sum) if previous_4_revenue_sum != 0 else (latest_4_revenue_sum > 0)
        profit_increased = (latest_4_profit_sum > previous_4_profit_sum) if previous_4_profit_sum != 0 else (latest_4_profit_sum > 0)

        if previous_4_revenue_sum == 0 and latest_4_revenue_sum == 0: revenue_increased = False
        if previous_4_profit_sum == 0 and latest_4_profit_sum == 0: profit_increased = False
        
        if revenue_increased and profit_increased: return '营收和净利均增长'
        elif revenue_increased and not profit_increased: return '营收增长净利下降'
        elif not revenue_increased and profit_increased: return '营收下降净利增长'
        else: return '营收和净利均下降'
    except Exception as e: 
        return '错误'

# ==============================================================================
#                                核心逻辑函数
# ==============================================================================

def run_script1_logic(today_pd, last_trade_day, df_all_stock_basic_info, df_latest_daily_data, df_latest_daily_basic_data):
    print("\n======================================================================")
    print("执行原脚本1的逻辑...")
    print("======================================================================\n")

    # 步骤 1：筛选换手率大于 2.5% 的股票
    print("脚本1 - 步骤1 - 筛选换手率...")
    df_merged_s1 = pd.merge(df_latest_daily_data, df_latest_daily_basic_data, on=['ts_code', 'trade_date'])
    turnover_rate_threshold = 2.5
    df_filtered_step1_s1 = df_merged_s1[df_merged_s1['turnover_rate'] > turnover_rate_threshold].copy()
    print(f"脚本1 - 步骤1 - 换手率筛选后剩余股票数量：{len(df_filtered_step1_s1)}")
    if df_filtered_step1_s1.empty:
        print("脚本1 - 没有股票满足换手率大于2.5%的条件，脚本1部分逻辑结束。")
        send_feishu_msg("脚本1结果", "无 (换手率筛选为空)")
        return

    ts_codes_step1_s1 = df_filtered_step1_s1['ts_code'].tolist()

    # 步骤 2：筛选收盘价上穿三条均线的股票
    print("脚本1 - 步骤2 - 计算MA并筛选...")
    ma_filtered_stocks_list_s1 = []
    start_date_ma_calc_s1 = (pd.Timestamp(last_trade_day) - pd.DateOffset(days=250)).strftime('%Y%m%d')

    for ts_code in tqdm(ts_codes_step1_s1, desc="脚本1 - 步骤2 处理股票MA"):
        stock_data_ma = get_stock_data_ma_cached(ts_code, start_date_ma_calc_s1, last_trade_day, ma_params=[20,60,120])
        if stock_data_ma is None or stock_data_ma.empty or len(stock_data_ma) < 120:
            continue
        
        last_row = stock_data_ma.iloc[-1]
        current_ma20, current_ma60, current_ma120 = last_row.get('ma20'), last_row.get('ma60'), last_row.get('ma120')

        if pd.isna(current_ma20) or pd.isna(current_ma60) or pd.isna(current_ma120):
            close_prices = stock_data_ma['close'].values
            if len(close_prices) >= 120:
                current_ma20 = talib.SMA(close_prices, timeperiod=20)[-1]
                current_ma60 = talib.SMA(close_prices, timeperiod=60)[-1]
                current_ma120 = talib.SMA(close_prices, timeperiod=120)[-1]
                if pd.isna(current_ma20) or pd.isna(current_ma60) or pd.isna(current_ma120):
                    continue
            else:
                continue
        
        if last_row['close'] > current_ma20 and \
           last_row['close'] > current_ma60 and \
           last_row['close'] > current_ma120:
            ma_filtered_stocks_list_s1.append({
                'ts_code': ts_code, 'close': last_row['close'],
                'ma20': current_ma20, 'ma60': current_ma60, 'ma120': current_ma120,
                'close_above_ma': True
            })
    
    ma_filtered_stocks_s1 = pd.DataFrame(ma_filtered_stocks_list_s1)
    print(f"脚本1 - 步骤2 - MA筛选后剩余股票数量：{len(ma_filtered_stocks_s1)}")
    if ma_filtered_stocks_s1.empty:
        print("脚本1 - 没有股票满足收盘价上穿三条均线的条件，脚本1部分逻辑结束。")
        send_feishu_msg("脚本1结果", "无 (均线筛选为空)")
        return

    # 步骤 3：获取股票基本信息和概念信息，并整合数据
    print("脚本1 - 步骤3 - 获取股票基本信息、概念、近期日线...")
    df_step3_merged_s1 = pd.merge(ma_filtered_stocks_s1, df_all_stock_basic_info[['ts_code', 'name', 'industry']], on='ts_code', how='left')
    
    df_result_so_far_s1 = df_step3_merged_s1.copy()
    df_result_so_far_s1['concept_name'] = 'N/A'
    df_result_so_far_s1['date'] = last_trade_day
    df_result_so_far_s1['change'] = np.nan
    df_result_so_far_s1['recent_limit_up'] = False
    df_result_so_far_s1['limit_up_days_in_last_5_days'] = 0

    start_date_5days_s1 = (pd.Timestamp(last_trade_day) - pd.DateOffset(days=10)).strftime('%Y%m%d')
    for index, row in tqdm(df_result_so_far_s1.iterrows(), total=df_result_so_far_s1.shape[0], desc="脚本1 - 步骤3 处理概念/日线"):
        ts_code = row['ts_code']
        concept_stock_info = get_concept_detail_cached(ts_code)
        if concept_stock_info is not None and not concept_stock_info.empty:
            df_result_so_far_s1.loc[index, 'concept_name'] = concept_stock_info['concept_name'].iloc[0]

        daily_data_5d = get_daily_data_for_period_cached(ts_code, start_date_5days_s1, last_trade_day)
        if daily_data_5d is not None and not daily_data_5d.empty:
            daily_data_5d = daily_data_5d.sort_values(by='trade_date', ascending=False).head(5)
            if not daily_data_5d.empty:
                latest_daily_row = daily_data_5d.iloc[0]
                if latest_daily_row['trade_date'] == last_trade_day:
                    df_result_so_far_s1.loc[index, 'change'] = latest_daily_row['pct_chg']
                    if pd.notna(latest_daily_row['pre_close']) and latest_daily_row['pre_close'] > 0:
                        limit_up_ratio = 0.10
                        stock_name_for_limit = row.get('name', '')
                        if stock_name_for_limit.startswith(('ST', '*ST')): limit_up_ratio = 0.05
                        limit_up_price = round(latest_daily_row['pre_close'] * (1 + limit_up_ratio), 2)
                        if latest_daily_row['close'] >= limit_up_price - 0.01 and latest_daily_row['close'] == latest_daily_row['high']:
                            df_result_so_far_s1.loc[index, 'recent_limit_up'] = True
                
                limit_up_days_count = 0
                for _, daily_row in daily_data_5d.iterrows():
                    if pd.notna(daily_row['pre_close']) and daily_row['pre_close'] > 0:
                        limit_up_ratio_hist = 0.10
                        if row.get('name','').startswith(('ST', '*ST')): limit_up_ratio_hist = 0.05
                        limit_up_price_hist = round(daily_row['pre_close'] * (1 + limit_up_ratio_hist), 2)
                        if daily_row['close'] >= limit_up_price_hist - 0.01 and daily_row['close'] == daily_row['high']:
                            limit_up_days_count += 1
                df_result_so_far_s1.loc[index, 'limit_up_days_in_last_5_days'] = limit_up_days_count
    print(f"脚本1 - 步骤3 - 数据收集完成，当前结果中共有 {len(df_result_so_far_s1)} 只股票的数据。")

    # 步骤 4：获取财务数据并计算 valuation_ratio
    print("脚本1 - 步骤4 - 获取财务数据并计算 valuation_ratio...")
    financial_details_list_s1 = []
    for ts_code in tqdm(df_result_so_far_s1['ts_code'].tolist(), desc="脚本1 - 步骤4 获取财务数据"):
        f_data = get_recent_financials_cached(ts_code)
        if f_data is not None:
            financial_details_list_s1.append(f_data)
    
    df_recent_financials_s1 = pd.DataFrame(financial_details_list_s1)

    if df_recent_financials_s1.empty:
        print("脚本1 - 没有获取到有效的财务数据，部分财务相关筛选可能无法进行。")
        financials_cols_expected = ['revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio']
        for col in financials_cols_expected:
            if col not in df_result_so_far_s1.columns: df_result_so_far_s1[col] = np.nan
    else:
        print(f"脚本1 - 获取到财务数据的股票数量：{len(df_recent_financials_s1)}")
        df_total_mv_subset_s1 = df_filtered_step1_s1[['ts_code', 'total_mv']].copy()
        df_total_mv_subset_s1.rename(columns={'total_mv': 'total_mv_finance'}, inplace=True)
        df_recent_financials_s1 = pd.merge(df_recent_financials_s1, df_total_mv_subset_s1, on='ts_code', how='left')

        df_recent_financials_s1['valuation_ratio'] = np.nan
        valid_calc_mask = (df_recent_financials_s1['revenue'].notna()) & \
                          (df_recent_financials_s1['deducted_net_profit_margin'].notna()) & \
                          (df_recent_financials_s1['total_mv_finance'].notna()) & \
                          (df_recent_financials_s1['total_mv_finance'] != 0) & \
                          (df_recent_financials_s1['revenue'] != 0)
        
        df_recent_financials_s1.loc[valid_calc_mask, 'valuation_ratio'] = (
            df_recent_financials_s1.loc[valid_calc_mask, 'revenue'] *
            (df_recent_financials_s1.loc[valid_calc_mask, 'deducted_net_profit_margin'] / 0.14) * 10
        ) / (df_recent_financials_s1.loc[valid_calc_mask, 'total_mv_finance'] * 10000)

        financial_cols_to_merge = ['ts_code', 'revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio']
        actual_financial_cols_to_merge = [col for col in financial_cols_to_merge if col in df_recent_financials_s1.columns]
        df_result_so_far_s1 = pd.merge(df_result_so_far_s1, df_recent_financials_s1[actual_financial_cols_to_merge], on='ts_code', how='left')

    # 步骤 5：计算抵扣价
    print("脚本1 - 步骤5 - 计算抵扣价...")
    df_result_so_far_s1['deduction_condition_met'] = False
    df_result_so_far_s1['抵扣价20'] = np.nan
    df_result_so_far_s1['抵扣价60'] = np.nan
    df_result_so_far_s1['抵扣价120'] = np.nan

    for index, row_main in tqdm(df_result_so_far_s1.iterrows(), total=df_result_so_far_s1.shape[0], desc="脚本1 - 步骤5 处理抵扣价"):
        ts_code = row_main['ts_code']
        stock_data_deduction = get_stock_data_ma_cached(ts_code, start_date_ma_calc_s1, last_trade_day, ma_params=[]) 
        if stock_data_deduction is None or stock_data_deduction.empty or len(stock_data_deduction) < 121:
            continue
        
        current_close_price = stock_data_deduction['close'].iloc[-1]
        try:
            deduction_20 = stock_data_deduction['close'].iloc[-21] if len(stock_data_deduction) >= 21 else np.nan
            deduction_60 = stock_data_deduction['close'].iloc[-61] if len(stock_data_deduction) >= 61 else np.nan
            deduction_120 = stock_data_deduction['close'].iloc[-121] if len(stock_data_deduction) >= 121 else np.nan

            df_result_so_far_s1.loc[index, '抵扣价20'] = deduction_20
            df_result_so_far_s1.loc[index, '抵扣价60'] = deduction_60
            df_result_so_far_s1.loc[index, '抵扣价120'] = deduction_120

            if pd.notna(deduction_20) and pd.notna(deduction_60) and pd.notna(deduction_120) and pd.notna(current_close_price):
                condition = (
                    (deduction_20 < 1.2 * current_close_price) and
                    (deduction_60 < 1.2 * current_close_price) and
                    (deduction_120 < 1.2 * current_close_price)
                )
                df_result_so_far_s1.loc[index, 'deduction_condition_met'] = condition
        except IndexError: pass
    print(f"脚本1 - 步骤5 - 满足抵扣价条件的股票数量：{df_result_so_far_s1['deduction_condition_met'].sum()}")

    # 步骤 6：筛选前一交易日收盘价小于等于前一日三条均线的股票
    print("脚本1 - 步骤6 - 前一日收盘价与均线关系...")
    df_result_so_far_s1['prev_day_close_vs_ma_condition'] = False
    for index, row_main in tqdm(df_result_so_far_s1.iterrows(), total=df_result_so_far_s1.shape[0], desc="脚本1 - 步骤6 处理MA额外条件"):
        ts_code = row_main['ts_code']
        stock_data_prev_ma = get_stock_data_ma_cached(ts_code, start_date_ma_calc_s1, last_trade_day, ma_params=[20,60,120])
        if stock_data_prev_ma is None or stock_data_prev_ma.empty or len(stock_data_prev_ma) < 121 or len(stock_data_prev_ma) < 2:
            continue
        
        prev_row = stock_data_prev_ma.iloc[-2]
        ma20_prev, ma60_prev, ma120_prev = prev_row.get('ma20'), prev_row.get('ma60'), prev_row.get('ma120')

        if pd.isna(prev_row['close']) or pd.isna(ma20_prev) or pd.isna(ma60_prev) or pd.isna(ma120_prev):
            continue
        
        if (prev_row['close'] <= ma20_prev) or \
           (prev_row['close'] <= ma60_prev) or \
           (prev_row['close'] <= ma120_prev):
            df_result_so_far_s1.loc[index, 'prev_day_close_vs_ma_condition'] = True
    print(f"脚本1 - 步骤6 - 满足额外MA条件股票数量：{df_result_so_far_s1['prev_day_close_vs_ma_condition'].sum()}")

    # 步骤 7：严苛财务和市场筛选
    print("脚本1 - 步骤7 - 严苛财务和市场筛选...")
    df_strict_filter_base_s1 = df_result_so_far_s1[df_result_so_far_s1['prev_day_close_vs_ma_condition']].copy()
    print(f"脚本1 - 步骤7 - 进入严苛筛选的股票数量：{len(df_strict_filter_base_s1)}")

    final_filtered_df_s1 = pd.DataFrame() # Initialize
    if df_strict_filter_base_s1.empty:
        print("脚本1 - 步骤7 - 没有股票满足进入严苛筛选的条件。")
    else:
        # 7.1: 扣非净利润率 > 14%
        print("脚本1 - 步骤7.1 - 筛选扣非净利润率...")
        if 'deducted_net_profit_margin' in df_strict_filter_base_s1.columns:
            df_step7_profit_margin_ok_s1 = df_strict_filter_base_s1[
                (df_strict_filter_base_s1['deducted_net_profit_margin'].notna()) &
                (df_strict_filter_base_s1['deducted_net_profit_margin'] > 0.14)
            ].copy()
        else:
            df_step7_profit_margin_ok_s1 = pd.DataFrame()
        print(f"脚本1 - 步骤7.1 - 满足扣非净利润率条件的股票数量：{len(df_step7_profit_margin_ok_s1)}")

        # 7.2: 估值比 > 1
        print("脚本1 - 步骤7.2 - 筛选估值比...")
        if 'valuation_ratio' in df_step7_profit_margin_ok_s1.columns:
            df_step7_valuation_ok_s1 = df_step7_profit_margin_ok_s1[
                (df_step7_profit_margin_ok_s1['valuation_ratio'].notna()) &
                (df_step7_profit_margin_ok_s1['valuation_ratio'] > 1)
            ].copy()
        else:
             df_step7_valuation_ok_s1 = pd.DataFrame()
        print(f"脚本1 - 步骤7.2 - 同时满足利润率和估值比的股票数量：{len(df_step7_valuation_ok_s1)}")

        # 7.3: 户均流通市值 > 15万
        print("脚本1 - 步骤7.3 - 筛选户均流通市值...")
        df_step7_market_value_ok_s1 = pd.DataFrame()
        if df_step7_valuation_ok_s1.empty:
            print("脚本1 - 步骤7.3 - 没有股票可进行户均流通市值筛选。")
        else:
            one_year_ago_for_holders_s1 = (pd.Timestamp(last_trade_day) - timedelta(days=365)).strftime('%Y%m%d')
            list_for_market_value_df_s1 = []
            for _, row_mv in tqdm(df_step7_valuation_ok_s1.iterrows(), total=df_step7_valuation_ok_s1.shape[0], desc="脚本1 - 步骤7.3 处理户均市值"):
                ts_code_mv = row_mv['ts_code']
                circulating_shares_val = np.nan
                num_shareholders_val = np.nan
                top10_total_shares = 0.0
                latest_close_price_mv = row_mv['close'] 

                daily_basic_info = get_daily_basic_cached(ts_code_mv, last_trade_day)
                if daily_basic_info is not None and not daily_basic_info.empty and pd.notna(daily_basic_info['float_share'].iloc[0]):
                    circulating_shares_val = daily_basic_info['float_share'].iloc[0] * 10000 
                
                holder_info = get_stk_holdernumber_cached(ts_code_mv, one_year_ago_for_holders_s1, last_trade_day)
                if holder_info is not None and not holder_info.empty:
                    holder_info = holder_info.sort_values(by='end_date', ascending=False)
                    if pd.notna(holder_info['holder_num'].iloc[0]):
                        num_shareholders_val = holder_info['holder_num'].iloc[0]

                top10_info = get_top10_floatholders_cached(ts_code_mv, one_year_ago_for_holders_s1, last_trade_day)
                if top10_info is not None and not top10_info.empty:
                    top10_info = top10_info.sort_values(by=['end_date', 'ann_date'], ascending=[False, False])
                    latest_report_end_date = top10_info['end_date'].iloc[0]
                    latest_top10_df = top10_info[top10_info['end_date'] == latest_report_end_date]
                    current_top10_sum = latest_top10_df['hold_amount'].sum()
                    if pd.notna(current_top10_sum): top10_total_shares = current_top10_sum
                
                calculated_mv_per_capita = np.nan
                if pd.notna(circulating_shares_val) and pd.notna(latest_close_price_mv) and \
                   pd.notna(num_shareholders_val) and num_shareholders_val > 0 and latest_close_price_mv > 0:
                    circulating_market_cap = circulating_shares_val * latest_close_price_mv
                    top10_market_cap = top10_total_shares * latest_close_price_mv
                    numerator = circulating_market_cap - top10_market_cap
                    denominator = num_shareholders_val - 10 if num_shareholders_val > 10 else num_shareholders_val
                    if denominator > 0: calculated_mv_per_capita = numerator / denominator
                    elif numerator == 0 and denominator == 0: calculated_mv_per_capita = 0
                
                if pd.notna(calculated_mv_per_capita) and calculated_mv_per_capita > 150000:
                    new_row = row_mv.copy()
                    new_row['per_capita_circulating_mv'] = calculated_mv_per_capita
                    list_for_market_value_df_s1.append(new_row)
            
            if list_for_market_value_df_s1:
                df_step7_market_value_ok_s1 = pd.DataFrame(list_for_market_value_df_s1).reset_index(drop=True)
        
        print(f"脚本1 - 步骤7.3 - 满足户均流通市值条件的股票数量：{len(df_step7_market_value_ok_s1)}")
        final_filtered_df_s1 = df_step7_market_value_ok_s1.copy()

        # 7.4: 增长情况分类
        print("脚本1 - 步骤7.4 - 增长情况分类...")
        if not final_filtered_df_s1.empty:
            final_filtered_df_s1['growth_classification'] = 'N/A'
            for index_gc in tqdm(final_filtered_df_s1.index, desc="脚本1 - 步骤7.4 分类增长"):
                ts_code_growth = final_filtered_df_s1.loc[index_gc, 'ts_code']
                classification = classify_growth_detail_s1(ts_code_growth) 
                final_filtered_df_s1.loc[index_gc, 'growth_classification'] = classification
        else:
            print("脚本1 - 步骤7.4 - 没有股票可进行增长分类。")

    # 步骤 8：行业统计
    print("脚本1 - 步骤8 - 行业统计...")
    if not df_result_so_far_s1.empty and 'industry' in df_result_so_far_s1.columns:
        df_result_so_far_s1['industry'].fillna('未知行业', inplace=True)
        industry_counts_s1 = df_result_so_far_s1['industry'].value_counts().reset_index()
        industry_counts_s1.columns = ['industry', 'stock_count_in_industry']
        df_result_so_far_s1 = df_result_so_far_s1.merge(industry_counts_s1, on='industry', how='left')
        if 'stock_count_in_industry' in df_result_so_far_s1.columns:
             df_result_so_far_s1 = df_result_so_far_s1.sort_values(by='stock_count_in_industry', ascending=False)
    else:
        if 'stock_count_in_industry' not in df_result_so_far_s1.columns: 
            df_result_so_far_s1['stock_count_in_industry'] = np.nan

    # 保存结果
    current_date_filename_s1 = pd.Timestamp(last_trade_day).strftime('%Y%m%d')
    general_result_filename_s1 = f'合并脚本输出_初步筛选完整结果_脚本1_{current_date_filename_s1}.csv'
    df_result_so_far_s1.to_csv(general_result_filename_s1, index=False, encoding='utf-8-sig')
    print(f"脚本1 - 初步筛选结果已保存到：{general_result_filename_s1}")

    strict_filtered_filename_s1 = f'合并脚本输出_严苛筛选最终结果_脚本1_{current_date_filename_s1}.csv'
    expected_final_columns_s1 = [
        'ts_code', 'name', 'industry', 'concept_name', 'date', 'close', 'change',
        'close_above_ma', 'recent_limit_up', 'limit_up_days_in_last_5_days', 'ma20', 'ma60', 'ma120',
        'revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio',
        'deduction_condition_met', '抵扣价20', '抵扣价60', '抵扣价120',
        'prev_day_close_vs_ma_condition',
        'per_capita_circulating_mv',
        'growth_classification'
    ]
    if final_filtered_df_s1.empty:
        final_filtered_df_s1 = pd.DataFrame(columns=expected_final_columns_s1)
    else:
        for col in expected_final_columns_s1:
            if col not in final_filtered_df_s1.columns:
                final_filtered_df_s1[col] = np.nan
        final_filtered_df_s1 = final_filtered_df_s1[expected_final_columns_s1]
    
    final_filtered_df_s1.to_csv(strict_filtered_filename_s1, index=False, encoding='utf-8-sig')
    print(f"脚本1 - 严苛筛选结果已保存到：{strict_filtered_filename_s1}")
    
    # --- 飞书发送集成 (在函数最后) ---
    print(">> 正在构建脚本1飞书消息...")
    msg_lines = []
    
    # 筛选条件: prev_day_close_vs_ma_condition为true，valuation_ratio>=1.3 (来自您的print逻辑)
    if not df_result_so_far_s1.empty:
        required_cols_gen_s1 = ['prev_day_close_vs_ma_condition', 'valuation_ratio', 'ts_code', 'name', 'industry']
        for r_col in required_cols_gen_s1:
            if r_col not in df_result_so_far_s1.columns: df_result_so_far_s1[r_col] = np.nan
        
        filtered_general_s1 = df_result_so_far_s1[
            (df_result_so_far_s1['prev_day_close_vs_ma_condition'] == True) &
            (df_result_so_far_s1['valuation_ratio'].notna()) &
            (df_result_so_far_s1['valuation_ratio'] >= 1.3)
        ]
        if not filtered_general_s1.empty:
            msg_lines.append("【初步筛选】(收盘<均线 & 估值>1.3):")
            for _, row in filtered_general_s1.head(10).iterrows(): # 限制展示前10
                msg_lines.append(f"{row['name']}({row['ts_code']}) {row['industry']} 估值:{row['valuation_ratio']:.2f}")
    
    # 严苛筛选结果
    if not final_filtered_df_s1.empty:
        msg_lines.append(f"\n【严苛筛选】(共{len(final_filtered_df_s1)}只):")
        for _, row in final_filtered_df_s1.head(20).iterrows():
            growth = row.get('growth_classification', 'N/A')
            msg_lines.append(f"{row['name']}({row['ts_code']}) 增长:{growth} 估值:{row.get('valuation_ratio', 0):.2f}")
    
    if msg_lines:
        send_feishu_msg("脚本1 筛选结果", "\n".join(msg_lines))
    else:
        send_feishu_msg("脚本1 筛选结果", "无符合条件标的")

    print(f"脚本1 - 逻辑执行完毕。共筛选出 {len(final_filtered_df_s1)} 只满足所有严苛条件的股票。")


def run_script2_logic(today_pd, last_trade_day, df_all_stock_basic_info, df_latest_daily_data, df_latest_daily_basic_data):
    print("\n======================================================================")
    print("执行原脚本2的逻辑...")
    print("======================================================================\n")

    # 步骤 1：构建涨停股票池
    print("脚本2 - 步骤1 - 构建涨停股票池...")
    num_trading_days_s2 = 2 
    
    time.sleep(API_CALL_DELAY)
    cal_s2 = pro.trade_cal(exchange='', start_date=(today_pd - timedelta(days=30)).strftime('%Y%m%d'), end_date=today_pd.strftime('%Y%m%d'))
    cal_s2 = cal_s2[cal_s2['is_open'] == 1].sort_values(by='cal_date', ascending=False).head(num_trading_days_s2)
    trade_dates_s2 = cal_s2['cal_date'].tolist()

    limit_up_counts_s2 = {}
    
    for trade_date in tqdm(trade_dates_s2, desc="脚本2 - 步骤1 获取涨停股票"):
        if trade_date == last_trade_day:
            daily_data_s2_loop = df_latest_daily_data.copy()
        else: 
            daily_data_s2_loop = get_daily_data_for_period_cached(None, trade_date, trade_date)
            if daily_data_s2_loop is None or daily_data_s2_loop.empty:
                 continue
        
        daily_data_s2_loop = pd.merge(daily_data_s2_loop, df_all_stock_basic_info[['ts_code', 'exchange', 'market']], on='ts_code', how='left')
        
        daily_data_s2_loop['limit_up_pct'] = daily_data_s2_loop.apply(
            lambda row: 10.0 if row['market'] in ['主板'] else (20.0 if row['market'] in ['科创板', '创业板'] else (30.0 if row['market'] == '北交所' else 10.0)),
            axis=1
        )
        daily_data_s2_loop['is_limit_up'] = (daily_data_s2_loop['pct_chg'] >= (daily_data_s2_loop['limit_up_pct'] - 0.1)) 
        limit_up_stocks_s2 = daily_data_s2_loop[daily_data_s2_loop['is_limit_up']]

        for ts_code in limit_up_stocks_s2['ts_code']:
            limit_up_counts_s2[ts_code] = limit_up_counts_s2.get(ts_code, 0) + 1
            
    stock_pool_s2 = pd.DataFrame(list(limit_up_counts_s2.items()), columns=['ts_code', 'limit_up_days'])
    stock_pool_s2 = pd.merge(stock_pool_s2, df_all_stock_basic_info[['ts_code', 'name', 'exchange', 'market', 'list_status']], on='ts_code', how='left')
    
    print(f"脚本2 - 步骤1 - 涨停股票池中股票数量：{len(stock_pool_s2)}")
    if stock_pool_s2.empty:
        print("脚本2 - 涨停股票池为空，脚本2逻辑结束。")
        send_feishu_msg("脚本2结果", "无 (涨停股票池为空)")
        return

    # 步骤 2：计算换手率
    print("脚本2 - 步骤2 - 计算换手率...")
    if not df_latest_daily_data.empty and not df_latest_daily_basic_data.empty:
        df_merged_turnover_s2 = pd.merge(df_latest_daily_data, df_latest_daily_basic_data, on=['ts_code', 'trade_date'], how='inner')
        df_merged_turnover_s2 = df_merged_turnover_s2[['ts_code', 'turnover_rate']]
        stock_pool_s2 = pd.merge(stock_pool_s2, df_merged_turnover_s2, on='ts_code', how='left')
    else:
        stock_pool_s2['turnover_rate'] = np.nan
    
    turnover_rate_threshold_s2 = 2.5
    stock_pool_s2['turnover_rate_pass'] = stock_pool_s2['turnover_rate'] > turnover_rate_threshold_s2
    print("脚本2 - 步骤2 - 换手率计算完成。")

    # 步骤 3：计算收盘价与三条均线的关系 (使用talib)
    print("脚本2 - 步骤3 - 计算MA并判断...")
    stock_pool_s2['close'] = np.nan
    stock_pool_s2['ma20'] = np.nan
    stock_pool_s2['ma60'] = np.nan
    stock_pool_s2['ma120'] = np.nan
    stock_pool_s2['ma_pass'] = False
    
    start_date_ma_calc_s2 = (pd.Timestamp(last_trade_day) - pd.DateOffset(days=200)).strftime('%Y%m%d') 

    for idx, row in tqdm(stock_pool_s2.iterrows(), total=stock_pool_s2.shape[0], desc="脚本2 - 步骤3 处理MA"):
        ts_code = row['ts_code']
        stock_data_ma = get_stock_data_talib_ma_cached(ts_code, start_date_ma_calc_s2, last_trade_day, ma_params=[20,60,120])
        if stock_data_ma is None or stock_data_ma.empty or len(stock_data_ma) < 120 : 
             continue
        
        last_row = stock_data_ma.iloc[-1]
        ma20, ma60, ma120 = last_row.get('ma20'), last_row.get('ma60'), last_row.get('ma120')
        if pd.isna(ma20) or pd.isna(ma60) or pd.isna(ma120):
            continue
            
        stock_pool_s2.at[idx, 'close'] = last_row['close']
        stock_pool_s2.at[idx, 'ma20'] = ma20
        stock_pool_s2.at[idx, 'ma60'] = ma60
        stock_pool_s2.at[idx, 'ma120'] = ma120
        
        if last_row['close'] > ma20 and last_row['close'] > ma60 and last_row['close'] > ma120:
            stock_pool_s2.at[idx, 'ma_pass'] = True
            
    print("脚本2 - 步骤3 - 均线判断完成。")

    # 步骤 4：计算扣非净利润率
    print("脚本2 - 步骤4 - 获取财务数据计算扣非净利润率...")
    financial_details_list_s2 = []
    for ts_code in tqdm(stock_pool_s2['ts_code'].tolist(), desc="脚本2 - 步骤4 获取财务"):
        f_data = get_recent_financials_cached(ts_code) 
        if f_data is not None:
            financial_details_list_s2.append(f_data.to_frame().T) 
            
    if financial_details_list_s2:
        recent_financials_data_s2 = pd.concat(financial_details_list_s2, ignore_index=True)
        cols_to_merge_s2 = ['ts_code', 'revenue', 'profit_dedt', 'deducted_net_profit_margin']
        actual_cols_to_merge_s2 = [col for col in cols_to_merge_s2 if col in recent_financials_data_s2.columns]
        stock_pool_s2 = pd.merge(stock_pool_s2, recent_financials_data_s2[actual_cols_to_merge_s2], on='ts_code', how='left')
    else:
        stock_pool_s2['revenue'] = np.nan
        stock_pool_s2['profit_dedt'] = np.nan
        stock_pool_s2['deducted_net_profit_margin'] = np.nan

    stock_pool_s2['deducted_net_profit_margin_pass'] = stock_pool_s2['deducted_net_profit_margin'] > 0.14
    print("脚本2 - 步骤4 - 扣非净利润率判断完成。")

    # 步骤 5：计算估值比
    print("脚本2 - 步骤5 - 计算估值比...")
    if not df_latest_daily_basic_data.empty:
        df_total_mv_s2 = df_latest_daily_basic_data[['ts_code', 'total_mv']]
        stock_pool_s2 = pd.merge(stock_pool_s2, df_total_mv_s2, on='ts_code', how='left')
    else:
        stock_pool_s2['total_mv'] = np.nan

    cols_for_valuation_s2 = ['revenue', 'deducted_net_profit_margin', 'total_mv']
    for col in cols_for_valuation_s2:
        if col not in stock_pool_s2.columns: stock_pool_s2[col] = np.nan
        stock_pool_s2[col] = pd.to_numeric(stock_pool_s2[col], errors='coerce')

    stock_pool_s2['valuation_ratio'] = (
        stock_pool_s2['revenue'] *
        (stock_pool_s2['deducted_net_profit_margin'] / 0.14) * 10
    ) / (stock_pool_s2['total_mv'] * 10000)
    stock_pool_s2['valuation_ratio_pass'] = stock_pool_s2['valuation_ratio'] > 1
    print("脚本2 - 步骤5 - 估值比计算完成。")

    # 步骤 6：计算户均流通市值
    print("脚本2 - 步骤6 - 计算户均流通市值...")
    stock_pool_s2['circulating_shares'] = np.nan 
    stock_pool_s2['top10_shares'] = np.nan
    stock_pool_s2['num_shareholders'] = np.nan
    stock_pool_s2['value_per_capita'] = np.nan 
    stock_pool_s2['value_pass'] = False
    
    one_year_ago_s2 = (pd.Timestamp(last_trade_day) - timedelta(days=365)).strftime('%Y%m%d')

    for idx, row in tqdm(stock_pool_s2.iterrows(), total=stock_pool_s2.shape[0], desc="脚本2 - 步骤6 处理户均市值"):
        ts_code = row['ts_code']
        latest_close = row.get('close') 
        if pd.isna(latest_close):
            latest_daily_for_stock = df_latest_daily_data[df_latest_daily_data['ts_code'] == ts_code]
            if not latest_daily_for_stock.empty:
                latest_close = latest_daily_for_stock['close'].iloc[0]
        if pd.isna(latest_close): continue

        circulating_shares = np.nan
        db_info = df_latest_daily_basic_data[df_latest_daily_basic_data['ts_code'] == ts_code]
        if not db_info.empty and pd.notna(db_info['float_share'].iloc[0]):
            circulating_shares = db_info['float_share'].iloc[0] * 10000
        if pd.isna(circulating_shares): 
            db_info_cached = get_daily_basic_cached(ts_code, last_trade_day)
            if db_info_cached is not None and not db_info_cached.empty and pd.notna(db_info_cached['float_share'].iloc[0]):
                 circulating_shares = db_info_cached['float_share'].iloc[0] * 10000
        if pd.isna(circulating_shares): continue

        num_shareholders = np.nan
        holder_data = get_stk_holdernumber_cached(ts_code, one_year_ago_s2, last_trade_day)
        if holder_data is not None and not holder_data.empty:
            holder_data = holder_data.sort_values(by='end_date', ascending=False)
            if not holder_data.empty:
                num_shareholders = holder_data['holder_num'].iloc[0]
        if pd.isna(num_shareholders): continue
        
        top10_shares = 0.0
        top10_data = get_top10_floatholders_cached(ts_code, one_year_ago_s2, last_trade_day)
        if top10_data is not None and not top10_data.empty:
            top10_data = top10_data.sort_values(by=['end_date','ann_date'], ascending=[False,False])
            if not top10_data.empty:
                latest_end_date_top10 = top10_data['end_date'].iloc[0]
                top10_shares = top10_data[top10_data['end_date'] == latest_end_date_top10]['hold_amount'].sum()
        
        value_calc = np.nan
        if pd.notna(circulating_shares) and pd.notna(latest_close) and pd.notna(num_shareholders):
            denominator = max(num_shareholders - 10, 1)
            if denominator > 0:
                value_calc = (circulating_shares * latest_close - top10_shares * latest_close) / denominator
        
        stock_pool_s2.at[idx, 'circulating_shares'] = circulating_shares
        stock_pool_s2.at[idx, 'top10_shares'] = top10_shares
        stock_pool_s2.at[idx, 'num_shareholders'] = num_shareholders
        stock_pool_s2.at[idx, 'value_per_capita'] = value_calc
        if pd.notna(value_calc) and value_calc > 150000:
            stock_pool_s2.at[idx, 'value_pass'] = True
            
    print("脚本2 - 步骤6 - 户均流通市值计算完成。")

    # 步骤 7：计算抵扣价
    print("脚本2 - 步骤7 - 计算抵扣价...")
    stock_pool_s2['deduction_20'] = np.nan
    stock_pool_s2['deduction_60'] = np.nan
    stock_pool_s2['deduction_120'] = np.nan
    stock_pool_s2['deduction_condition_met'] = False
    
    for idx, row in tqdm(stock_pool_s2.iterrows(), total=stock_pool_s2.shape[0], desc="脚本2 - 步骤7 处理抵扣价"):
        ts_code = row['ts_code']
        stock_data_deduction = get_stock_data_talib_ma_cached(ts_code, start_date_ma_calc_s2, last_trade_day, ma_params=[]) 
        if stock_data_deduction is None or stock_data_deduction.empty: continue

        last_day_close = row.get('close') 
        if pd.isna(last_day_close) and not stock_data_deduction.empty:
             last_day_close = stock_data_deduction['close'].iloc[-1]
        if pd.isna(last_day_close): continue
            
        try:
            deduction_20 = stock_data_deduction['close'].iloc[-21] if len(stock_data_deduction) >= 21 else np.nan
            deduction_60 = stock_data_deduction['close'].iloc[-61] if len(stock_data_deduction) >= 61 else np.nan
            deduction_120 = stock_data_deduction['close'].iloc[-121] if len(stock_data_deduction) >= 121 else np.nan
        except IndexError:
            deduction_20, deduction_60, deduction_120 = np.nan, np.nan, np.nan

        stock_pool_s2.at[idx, 'deduction_20'] = deduction_20
        stock_pool_s2.at[idx, 'deduction_60'] = deduction_60
        stock_pool_s2.at[idx, 'deduction_120'] = deduction_120

        if pd.notna(deduction_20) and pd.notna(deduction_60) and pd.notna(deduction_120) and pd.notna(last_day_close):
            condition = (
                (deduction_20 < 1.2 * last_day_close) and
                (deduction_60 < 1.2 * last_day_close) and
                (deduction_120 < 1.2 * last_day_close)
            )
            stock_pool_s2.at[idx, 'deduction_condition_met'] = condition
    print("脚本2 - 步骤7 - 抵扣价判断完成。")

    # 步骤 8：判断前一交易日收盘价与MA的关系
    print("脚本2 - 步骤8 - 前一日收盘价与均线关系...")
    stock_pool_s2['previous_close_below_ma'] = False
    for idx, row in tqdm(stock_pool_s2.iterrows(), total=stock_pool_s2.shape[0], desc="脚本2 - 步骤8 处理前日MA"):
        ts_code = row['ts_code']
        stock_data_prev_ma = get_stock_data_talib_ma_cached(ts_code, start_date_ma_calc_s2, last_trade_day, ma_params=[20,60,120])
        if stock_data_prev_ma is None or stock_data_prev_ma.empty or len(stock_data_prev_ma) < 121 or len(stock_data_prev_ma) < 2:
            continue
        
        prev_row = stock_data_prev_ma.iloc[-2]
        ma20_prev, ma60_prev, ma120_prev = prev_row.get('ma20'), prev_row.get('ma60'), prev_row.get('ma120')
        if pd.isna(prev_row['close']) or pd.isna(ma20_prev) or pd.isna(ma60_prev) or pd.isna(ma120_prev):
            continue
        
        if (prev_row['close'] <= ma20_prev) or \
           (prev_row['close'] <= ma60_prev) or \
           (prev_row['close'] <= ma120_prev):
            stock_pool_s2.at[idx, 'previous_close_below_ma'] = True
    print("脚本2 - 步骤8 - 判断完成。")

    # 步骤 9：增长情况分类
    print("脚本2 - 步骤9 - 增长情况分类...")
    stock_pool_s2['growth_classification'] = 'N/A'
    for idx, row in tqdm(stock_pool_s2.iterrows(), total=stock_pool_s2.shape[0], desc="脚本2 - 步骤9 分类增长"):
        ts_code = row['ts_code']
        classification = classify_growth_detail_s1(ts_code) 
        stock_pool_s2.at[idx, 'growth_classification'] = classification
    print("脚本2 - 步骤9 - 分类完成。")

    # 步骤 10：行业统计
    print("脚本2 - 步骤10 - 行业统计...")
    if 'industry' not in stock_pool_s2.columns: 
        stock_pool_s2 = pd.merge(stock_pool_s2, df_all_stock_basic_info[['ts_code', 'industry']], on='ts_code', how='left')
    
    stock_pool_s2['industry'].fillna('未知行业', inplace=True)
    industry_counts_s2 = stock_pool_s2['industry'].value_counts().reset_index()
    industry_counts_s2.columns = ['industry', 'industry_stock_count']
    stock_pool_s2 = pd.merge(stock_pool_s2, industry_counts_s2, on='industry', how='left')
    print("脚本2 - 步骤10 - 行业统计完成。")

    # 保存最终结果
    current_date_s2 = dt.today().strftime('%Y%m%d') 
    final_file_name_s2 = f'合并脚本输出_涨停股票筛选结果_脚本2_{current_date_s2}.csv'
    stock_pool_s2.to_csv(final_file_name_s2, index=False, encoding='utf-8-sig')
    print(f"脚本2 - 最终结果已保存到：{final_file_name_s2}")

    # --- 飞书发送集成 (基于步骤11的逻辑) ---
    print(">> 正在构建脚本2飞书消息...")
    
    # 逻辑复刻自您的步骤11: previous_close_below_ma == True 且 valuation_ratio >= 1.3
    # 注意：stock_pool_s2 即为当前所有数据的集合，直接筛选即可
    
    if 'previous_close_below_ma' in stock_pool_s2.columns:
        # 确保布尔类型正确
        stock_pool_s2['previous_close_below_ma'] = stock_pool_s2['previous_close_below_ma'].astype(bool)
    
    condition1_s2 = stock_pool_s2['previous_close_below_ma'] == True
    condition2_s2 = stock_pool_s2['valuation_ratio'] >= 1.3
    filtered_stocks_s2 = stock_pool_s2[condition1_s2 & condition2_s2].copy()
    
    msg_lines = []
    if not filtered_stocks_s2.empty:
        msg_lines.append(f"【涨停回踩筛选】(共{len(filtered_stocks_s2)}只):")
        # 排序优化展示
        filtered_stocks_s2 = filtered_stocks_s2.sort_values(by='valuation_ratio', ascending=False)
        for _, row in filtered_stocks_s2.head(20).iterrows():
            growth = row.get('growth_classification', 'N/A')
            msg_lines.append(f"{row.get('name','N/A')}({row['ts_code']}) 增长:{growth} 估值:{row.get('valuation_ratio',0):.2f}")
        if len(filtered_stocks_s2) > 20:
             msg_lines.append(f"...等共{len(filtered_stocks_s2)}只")
    else:
        msg_lines.append("无满足条件 (收盘<均线 & 估值>1.3) 的股票")

    send_feishu_msg("脚本2 筛选结果", "\n".join(msg_lines))
    print(f"脚本2 - 逻辑执行完毕。")


def run_script3_logic(today_pd, last_trade_day, df_all_stock_basic_info, df_latest_daily_data, df_latest_daily_basic_data):
    print("\n======================================================================")
    print("执行原脚本3的逻辑...")
    print("======================================================================\n")

    # 步骤 1：筛选换手率并获取基本信息
    print("脚本3 - 步骤1 - 筛选换手率...")
    df_merged_s3 = pd.merge(df_latest_daily_data, df_latest_daily_basic_data, on=['ts_code', 'trade_date'])
    turnover_rate_threshold_s3 = 2.5
    df_filtered_step1_raw_s3 = df_merged_s3[df_merged_s3['turnover_rate'] > turnover_rate_threshold_s3].copy()
    print(f"脚本3 - 步骤1 - 换手率筛选后剩余股票数量：{len(df_filtered_step1_raw_s3)}")
    if df_filtered_step1_raw_s3.empty:
        print("脚本3 - 没有股票满足换手率条件，脚本3逻辑结束。")
        send_feishu_msg("脚本3结果", "无 (换手率筛选为空)")
        return

    print("脚本3 - 步骤1.5 - 合并股票基本信息 (名称、行业)...")
    df_filtered_step1_with_names_s3 = pd.merge(df_filtered_step1_raw_s3, df_all_stock_basic_info[['ts_code', 'name', 'industry']], on='ts_code', how='left')
    if df_filtered_step1_with_names_s3.empty:
        return

    # 步骤 2：筛选股价大幅偏离均线的股票
    print("脚本3 - 步骤2 - 筛选股价大幅偏离均线的股票...")
    step2_list_for_df_s3 = []
    start_date_step2_calc_s3 = (pd.Timestamp(last_trade_day) - pd.DateOffset(days=250)).strftime('%Y%m%d')

    for _, row_step1 in tqdm(df_filtered_step1_with_names_s3.iterrows(), total=df_filtered_step1_with_names_s3.shape[0], desc="脚本3 - 步骤2 处理MA偏离"):
        ts_code = row_step1['ts_code']
        stock_name = row_step1.get('name', 'N/A')
        stock_industry = row_step1.get('industry', 'N/A')
        
        stock_data_s3 = get_stock_data_ma_cached(ts_code, start_date_step2_calc_s3, last_trade_day, ma_params=[20,60,120])
        if stock_data_s3 is None or stock_data_s3.empty or len(stock_data_s3) < 120:
            continue
            
        last_row = stock_data_s3.iloc[-1]
        current_high, current_low, current_close = last_row['high'], last_row['low'], last_row['close']
        current_ma20, current_ma60, current_ma120 = last_row.get('ma20'), last_row.get('ma60'), last_row.get('ma120')

        if pd.isna(current_ma20) or pd.isna(current_ma60) or pd.isna(current_ma120): 
            close_prices = stock_data_s3['close'].values
            if len(close_prices) >= 120:
                current_ma20 = talib.SMA(close_prices, timeperiod=20)[-1]
                current_ma60 = talib.SMA(close_prices, timeperiod=60)[-1]
                current_ma120 = talib.SMA(close_prices, timeperiod=120)[-1]
            else: continue
        if pd.isna(current_high) or pd.isna(current_low) or pd.isna(current_close) or \
           pd.isna(current_ma20) or pd.isna(current_ma60) or pd.isna(current_ma120): continue

        deviation_type = 0
        actual_deviation_pct = np.nan
        ma_values = [m for m in [current_ma20, current_ma60, current_ma120] if pd.notna(m)] 
        if not ma_values: continue 
        
        max_ma = max(ma_values)
        min_ma = min(ma_values)

        condition_up = (current_high > current_ma20 and current_high > current_ma60 and \
                        current_high > current_ma120 and max_ma > 0 and current_high >= max_ma * 1.15)
        condition_down = (current_low < current_ma20 and current_low < current_ma60 and \
                          current_low < current_ma120 and min_ma > 0 and current_low <= min_ma * 0.85)

        if condition_up:
            deviation_type = 1
            if max_ma > 0: actual_deviation_pct = (current_high / max_ma - 1) * 100
        elif condition_down:
            deviation_type = -1
            if min_ma > 0: actual_deviation_pct = (current_low / min_ma - 1) * 100
        
        if deviation_type != 0:
            step2_list_for_df_s3.append({
                'ts_code': ts_code, 'name': stock_name, 'industry': stock_industry,
                'close': current_close, 'high': current_high, 'low': current_low,
                'ma20': current_ma20, 'ma60': current_ma60, 'ma120': current_ma120,
                'ma_deviation_type': deviation_type, 'ma_actual_deviation_pct': actual_deviation_pct
            })
    
    ma_deviation_stocks_df_s3 = pd.DataFrame(step2_list_for_df_s3)
    print(f"脚本3 - 步骤2 - MA偏离筛选后剩余股票数量：{len(ma_deviation_stocks_df_s3)}")

    if ma_deviation_stocks_df_s3.empty:
        print("脚本3 - 没有股票满足步骤2的MA偏离条件，脚本3逻辑结束。")
        send_feishu_msg("脚本3结果", "无 (偏离筛选为空)")
        return
    
    # 步骤 3：获取概念信息等
    print("脚本3 - 步骤3 - 获取概念、近期日线...")
    df_result_so_far_s3 = ma_deviation_stocks_df_s3.copy()
    df_result_so_far_s3['concept_name'] = 'N/A'
    df_result_so_far_s3['date'] = last_trade_day
    df_result_so_far_s3['change'] = np.nan
    df_result_so_far_s3['recent_limit_up'] = False
    df_result_so_far_s3['limit_up_days_in_last_5_days'] = 0

    start_date_5days_s3 = (pd.Timestamp(last_trade_day) - pd.DateOffset(days=10)).strftime('%Y%m%d')
    for index, row in tqdm(df_result_so_far_s3.iterrows(), total=df_result_so_far_s3.shape[0], desc="脚本3 - 步骤3 处理概念/日线"):
        ts_code = row['ts_code']
        concept_info = get_concept_detail_cached(ts_code)
        if concept_info is not None and not concept_info.empty:
            df_result_so_far_s3.loc[index, 'concept_name'] = concept_info['concept_name'].iloc[0]
        
        daily_data_5d_s3 = get_daily_data_for_period_cached(ts_code, start_date_5days_s3, last_trade_day)
        if daily_data_5d_s3 is not None and not daily_data_5d_s3.empty:
            daily_data_5d_s3 = daily_data_5d_s3.sort_values(by='trade_date', ascending=False).head(5)
            if not daily_data_5d_s3.empty:
                latest_daily_row = daily_data_5d_s3.iloc[0]
                if latest_daily_row['trade_date'] == last_trade_day:
                    df_result_so_far_s3.loc[index, 'change'] = latest_daily_row['pct_chg']
                    if pd.notna(latest_daily_row['pre_close']) and latest_daily_row['pre_close'] > 0:
                        limit_up_ratio = 0.10
                        stock_name_for_limit = row.get('name', '')
                        if stock_name_for_limit.startswith(('ST', '*ST')): limit_up_ratio = 0.05
                        limit_up_price = round(latest_daily_row['pre_close'] * (1 + limit_up_ratio), 2)
                        if latest_daily_row['close'] >= limit_up_price - 0.01 and latest_daily_row['close'] == latest_daily_row['high']:
                            df_result_so_far_s3.loc[index, 'recent_limit_up'] = True
                
                limit_up_days_count = 0
                for _, daily_row_hist in daily_data_5d_s3.iterrows():
                    if pd.notna(daily_row_hist['pre_close']) and daily_row_hist['pre_close'] > 0:
                        limit_up_ratio_hist = 0.10
                        if row.get('name','').startswith(('ST', '*ST')): limit_up_ratio_hist = 0.05
                        limit_up_price_hist = round(daily_row_hist['pre_close'] * (1 + limit_up_ratio_hist), 2)
                        if daily_row_hist['close'] >= limit_up_price_hist -0.01 and daily_row_hist['close'] == daily_row_hist['high']:
                            limit_up_days_count +=1
                df_result_so_far_s3.loc[index, 'limit_up_days_in_last_5_days'] = limit_up_days_count
    print(f"脚本3 - 步骤3 - 数据收集完成，当前结果中共有 {len(df_result_so_far_s3)} 只股票。")
    
    # 步骤 4：获取财务数据并计算 valuation_ratio
    print("脚本3 - 步骤4 - 获取财务数据并计算 valuation_ratio...")
    financial_details_list_s3 = []
    for ts_code in tqdm(df_result_so_far_s3['ts_code'].tolist(), desc="脚本3 - 步骤4 获取财务"):
        f_data = get_recent_financials_cached(ts_code)
        if f_data is not None:
            financial_details_list_s3.append(f_data) 
            
    df_recent_financials_s3 = pd.DataFrame(financial_details_list_s3) 

    if df_recent_financials_s3.empty:
        print("脚本3 - 没有获取到有效财务数据。")
        financials_cols_expected = ['revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio']
        for col in financials_cols_expected:
            if col not in df_result_so_far_s3.columns: df_result_so_far_s3[col] = np.nan
    else:
        print(f"脚本3 - 获取到财务数据的股票数量：{len(df_recent_financials_s3)}")
        df_total_mv_subset_s3 = df_filtered_step1_raw_s3[['ts_code', 'total_mv']].copy()
        df_total_mv_subset_s3.rename(columns={'total_mv': 'total_mv_finance'}, inplace=True)
        df_recent_financials_s3 = pd.merge(df_recent_financials_s3, df_total_mv_subset_s3, on='ts_code', how='left')

        df_recent_financials_s3['valuation_ratio'] = np.nan
        valid_calc_mask_s3 = (df_recent_financials_s3['revenue'].notna()) & \
                             (df_recent_financials_s3['deducted_net_profit_margin'].notna()) & \
                             (df_recent_financials_s3['total_mv_finance'].notna()) & \
                             (df_recent_financials_s3['total_mv_finance'] != 0) & \
                             (df_recent_financials_s3['revenue'] != 0)
        df_recent_financials_s3.loc[valid_calc_mask_s3, 'valuation_ratio'] = (
            df_recent_financials_s3.loc[valid_calc_mask_s3, 'revenue'] *
            (df_recent_financials_s3.loc[valid_calc_mask_s3, 'deducted_net_profit_margin'] / 0.14) * 10
        ) / (df_recent_financials_s3.loc[valid_calc_mask_s3, 'total_mv_finance'] * 10000)
        
        financial_cols_to_merge_s3 = ['ts_code', 'revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio']
        actual_cols_s3 = [col for col in financial_cols_to_merge_s3 if col in df_recent_financials_s3.columns]
        df_result_so_far_s3 = pd.merge(df_result_so_far_s3, df_recent_financials_s3[actual_cols_s3], on='ts_code', how='left')

    # 步骤 7：严苛筛选
    print("脚本3 - 步骤7 - 严苛财务和市场筛选...")
    df_strict_filter_base_s3 = df_result_so_far_s3.copy() 
    print(f"脚本3 - 步骤7 - 进入严苛筛选的股票数量：{len(df_strict_filter_base_s3)}")
    
    final_filtered_df_s3 = pd.DataFrame() 
    if df_strict_filter_base_s3.empty:
        print("脚本3 - 步骤7 - 没有股票可进入严苛筛选。")
    else:
        # 7.1 扣非净利润率
        print("脚本3 - 步骤7.1 - 筛选扣非净利润率...")
        if 'deducted_net_profit_margin' in df_strict_filter_base_s3.columns:
            df_step7_profit_margin_ok_s3 = df_strict_filter_base_s3[
                (df_strict_filter_base_s3['deducted_net_profit_margin'].notna()) &
                (df_strict_filter_base_s3['deducted_net_profit_margin'] > 0.14)
            ].copy()
        else: df_step7_profit_margin_ok_s3 = pd.DataFrame()
        print(f"脚本3 - 步骤7.1 - 满足扣非净利润率条件的股票数量：{len(df_step7_profit_margin_ok_s3)}")
        
        # 7.2 估值比
        print("脚本3 - 步骤7.2 - 筛选估值比...")
        if 'valuation_ratio' in df_step7_profit_margin_ok_s3.columns:
            df_step7_valuation_ok_s3 = df_step7_profit_margin_ok_s3[
                (df_step7_profit_margin_ok_s3['valuation_ratio'].notna()) &
                (df_step7_profit_margin_ok_s3['valuation_ratio'] > 1)
            ].copy()
        else: df_step7_valuation_ok_s3 = pd.DataFrame()
        print(f"脚本3 - 步骤7.2 - 同时满足利润率和估值比的股票数量：{len(df_step7_valuation_ok_s3)}")

        # 7.3 户均流通市值
        print("脚本3 - 步骤7.3 - 筛选户均流通市值...")
        df_step7_market_value_ok_s3 = pd.DataFrame()
        if df_step7_valuation_ok_s3.empty:
            print("脚本3 - 步骤7.3 - 没有股票可进行户均流通市值筛选。")
        else:
            one_year_ago_s3 = (pd.Timestamp(last_trade_day) - timedelta(days=365)).strftime('%Y%m%d')
            list_for_mv_df_s3 = []
            for _, row_mv in tqdm(df_step7_valuation_ok_s3.iterrows(), total=df_step7_valuation_ok_s3.shape[0], desc="脚本3 - 步骤7.3 处理户均市值"):
                ts_code_mv = row_mv['ts_code']
                circulating_shares_val, num_shareholders_val = np.nan, np.nan
                top10_total_shares = 0.0
                latest_close_price_mv = row_mv['close']

                db_info_s3 = get_daily_basic_cached(ts_code_mv, last_trade_day)
                if db_info_s3 is not None and not db_info_s3.empty and pd.notna(db_info_s3['float_share'].iloc[0]):
                    circulating_shares_val = db_info_s3['float_share'].iloc[0] * 10000
                
                holder_info_s3 = get_stk_holdernumber_cached(ts_code_mv, one_year_ago_s3, last_trade_day)
                if holder_info_s3 is not None and not holder_info_s3.empty:
                    holder_info_s3 = holder_info_s3.sort_values(by='end_date', ascending=False)
                    if not holder_info_s3.empty and pd.notna(holder_info_s3['holder_num'].iloc[0]):
                        num_shareholders_val = holder_info_s3['holder_num'].iloc[0]
                
                top10_info_s3 = get_top10_floatholders_cached(ts_code_mv, one_year_ago_s3, last_trade_day)
                if top10_info_s3 is not None and not top10_info_s3.empty:
                    top10_info_s3 = top10_info_s3.sort_values(by=['end_date', 'ann_date'], ascending=[False, False])
                    if not top10_info_s3.empty:
                        latest_report_end_date_s3 = top10_info_s3['end_date'].iloc[0]
                        latest_top10_df_s3 = top10_info_s3[top10_info_s3['end_date'] == latest_report_end_date_s3]
                        current_top10_sum_s3 = latest_top10_df_s3['hold_amount'].sum()
                        if pd.notna(current_top10_sum_s3): top10_total_shares = current_top10_sum_s3

                calculated_mv_per_capita = np.nan
                if pd.notna(circulating_shares_val) and pd.notna(latest_close_price_mv) and \
                   pd.notna(num_shareholders_val) and num_shareholders_val > 0 and latest_close_price_mv > 0:
                    circulating_market_cap = circulating_shares_val * latest_close_price_mv
                    top10_market_cap = top10_total_shares * latest_close_price_mv
                    numerator = circulating_market_cap - top10_market_cap
                    denominator = num_shareholders_val - 10 if num_shareholders_val > 10 else num_shareholders_val
                    if denominator > 0: calculated_mv_per_capita = numerator / denominator
                    elif numerator == 0 and denominator == 0: calculated_mv_per_capita = 0
                
                if pd.notna(calculated_mv_per_capita) and calculated_mv_per_capita > 150000:
                    new_row = row_mv.copy()
                    new_row['per_capita_circulating_mv'] = calculated_mv_per_capita
                    list_for_mv_df_s3.append(new_row)
            
            if list_for_mv_df_s3:
                df_step7_market_value_ok_s3 = pd.DataFrame(list_for_mv_df_s3).reset_index(drop=True)
        
        print(f"脚本3 - 步骤7.3 - 满足户均流通市值条件的股票数量：{len(df_step7_market_value_ok_s3)}")
        final_filtered_df_s3 = df_step7_market_value_ok_s3.copy()

        # 7.4 增长分类
        print("脚本3 - 步骤7.4 - 增长情况分类...")
        if not final_filtered_df_s3.empty:
            final_filtered_df_s3['growth_classification'] = 'N/A'
            for index_gc in tqdm(final_filtered_df_s3.index, desc="脚本3 - 步骤7.4 分类增长"):
                ts_code_growth = final_filtered_df_s3.loc[index_gc, 'ts_code']
                classification = classify_growth_detail_s1(ts_code_growth) 
                final_filtered_df_s3.loc[index_gc, 'growth_classification'] = classification
        else:
            print("脚本3 - 步骤7.4 - 没有股票可进行增长分类。")
    
    # 步骤 8：行业统计
    print("脚本3 - 步骤8 - 行业统计...")
    if not df_result_so_far_s3.empty and 'industry' in df_result_so_far_s3.columns:
        df_result_so_far_s3['industry'].fillna('未知行业', inplace=True)
        industry_counts_s3 = df_result_so_far_s3['industry'].value_counts().reset_index()
        industry_counts_s3.columns = ['industry', 'stock_count_in_industry']
        df_result_so_far_s3 = df_result_so_far_s3.merge(industry_counts_s3, on='industry', how='left')
        if 'stock_count_in_industry' in df_result_so_far_s3.columns:
             df_result_so_far_s3 = df_result_so_far_s3.sort_values(by='stock_count_in_industry', ascending=False)
    else:
        if 'stock_count_in_industry' not in df_result_so_far_s3.columns:
             df_result_so_far_s3['stock_count_in_industry'] = np.nan

    # 保存结果
    current_date_filename_s3 = pd.Timestamp(last_trade_day).strftime('%Y%m%d')
    general_result_filename_s3 = f'合并脚本输出_极限偏离初步筛选完整结果_脚本3_{current_date_filename_s3}.csv'
    df_result_so_far_s3.to_csv(general_result_filename_s3, index=False, encoding='utf-8-sig')
    print(f"脚本3 - 初步筛选结果已保存到：{general_result_filename_s3}")

    strict_filtered_filename_s3 = f'合并脚本输出_极限偏离严苛筛选最终结果_脚本3_{current_date_filename_s3}.csv'
    expected_final_columns_s3 = [
        'ts_code', 'name', 'industry', 'concept_name', 'date',
        'close', 'high', 'low',
        'change', 'recent_limit_up', 'limit_up_days_in_last_5_days',
        'ma20', 'ma60', 'ma120', 'ma_deviation_type', 'ma_actual_deviation_pct',
        'revenue', 'profit_dedt', 'deducted_net_profit_margin', 'total_mv_finance', 'valuation_ratio',
        'per_capita_circulating_mv',
        'growth_classification',
        'stock_count_in_industry' 
    ]
    if final_filtered_df_s3.empty:
        final_filtered_df_s3 = pd.DataFrame(columns=expected_final_columns_s3)
    else:
        if 'stock_count_in_industry' in df_result_so_far_s3.columns and 'industry' in final_filtered_df_s3.columns:
            industry_counts_for_final_s3 = df_result_so_far_s3[['industry', 'stock_count_in_industry']].drop_duplicates(subset=['industry'])
            final_filtered_df_s3 = pd.merge(final_filtered_df_s3, industry_counts_for_final_s3, on='industry', how='left', suffixes=('', '_from_general'))
            if 'stock_count_in_industry_from_general' in final_filtered_df_s3.columns:
                final_filtered_df_s3['stock_count_in_industry'] = final_filtered_df_s3['stock_count_in_industry_from_general']
                final_filtered_df_s3.drop(columns=['stock_count_in_industry_from_general'], inplace=True)

        for col in expected_final_columns_s3:
            if col not in final_filtered_df_s3.columns:
                final_filtered_df_s3[col] = np.nan
        final_filtered_df_s3 = final_filtered_df_s3.reindex(columns=expected_final_columns_s3)
    
    final_filtered_df_s3.to_csv(strict_filtered_filename_s3, index=False, encoding='utf-8-sig')
    print(f"脚本3 - 严苛筛选结果已保存到：{strict_filtered_filename_s3}")
    
    # --- 飞书发送集成 (基于最终结果) ---
    print(">> 正在构建脚本3飞书消息...")
    msg_lines = []
    
    if not final_filtered_df_s3.empty:
        msg_lines.append(f"【极限偏离筛选】(共{len(final_filtered_df_s3)}只):")
        # 按偏离类型和幅度排序
        if 'ma_actual_deviation_pct' in final_filtered_df_s3.columns:
            final_filtered_df_s3['abs_dev'] = final_filtered_df_s3['ma_actual_deviation_pct'].abs()
            final_filtered_df_s3 = final_filtered_df_s3.sort_values(by='abs_dev', ascending=False)
            
        for _, row in final_filtered_df_s3.head(20).iterrows():
            dev_pct = row.get('ma_actual_deviation_pct', 0)
            dev_str = f"{dev_pct:.1f}%" if pd.notna(dev_pct) else "N/A"
            dev_type = row.get('ma_deviation_type', 0)
            type_icon = "📈" if dev_type == 1 else "📉"
            msg_lines.append(f"{row.get('name','N/A')}({row['ts_code']}) {type_icon} 幅度:{dev_str}")
            
        if len(final_filtered_df_s3) > 20:
             msg_lines.append(f"...等共{len(final_filtered_df_s3)}只")
    else:
        msg_lines.append("无满足极限偏离筛选条件的股票")
    
    send_feishu_msg("脚本3 筛选结果", "\n".join(msg_lines))
    print(f"脚本3 - 逻辑执行完毕。")


# --- 程序入口 ---
if __name__ == '__main__':
    start_time = time.time()
    today_pandas_timestamp = pd.Timestamp.today()
    
    try:
        LAST_TRADE_DAY = get_last_trade_day(today_pandas_timestamp)
        print("正在获取全市场股票基本信息...")
        time.sleep(API_CALL_DELAY)
        DF_ALL_STOCK_BASIC = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry,exchange,market,list_status')
        print(f"获取到 {len(DF_ALL_STOCK_BASIC)} 条股票基本信息。")

        print(f"正在获取最近交易日 ({LAST_TRADE_DAY}) 的全市场日线数据...")
        time.sleep(API_CALL_DELAY)
        DF_LATEST_DAILY = pro.daily(trade_date=LAST_TRADE_DAY)
        print(f"获取到 {len(DF_LATEST_DAILY)} 条当日日线数据。")
        
        print(f"正在获取最近交易日 ({LAST_TRADE_DAY}) 的全市场日线基本面数据...")
        time.sleep(API_CALL_DELAY)
        DF_LATEST_DAILY_BASIC = pro.daily_basic(trade_date=LAST_TRADE_DAY)
        print(f"获取到 {len(DF_LATEST_DAILY_BASIC)} 条当日日线基本面数据。")

        if DF_ALL_STOCK_BASIC.empty or DF_LATEST_DAILY.empty or DF_LATEST_DAILY_BASIC.empty:
            print("错误：未能获取到必要的全局市场数据，程序终止。")
            send_feishu_msg("严重错误", "未能获取到Tushare全局市场数据，请检查Token或网络。")
            exit()

        # 执行各脚本逻辑
        run_script1_logic(today_pandas_timestamp, LAST_TRADE_DAY, DF_ALL_STOCK_BASIC, DF_LATEST_DAILY, DF_LATEST_DAILY_BASIC)
        run_script2_logic(today_pandas_timestamp, LAST_TRADE_DAY, DF_ALL_STOCK_BASIC, DF_LATEST_DAILY, DF_LATEST_DAILY_BASIC)
        run_script3_logic(today_pandas_timestamp, LAST_TRADE_DAY, DF_ALL_STOCK_BASIC, DF_LATEST_DAILY, DF_LATEST_DAILY_BASIC)
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        send_feishu_msg("脚本运行出错", f"错误详情:\n{str(e)}")
        
    end_time = time.time()
    print("\n======================================================================")
    print("所有脚本逻辑已执行完毕。")
    print("======================================================================")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
