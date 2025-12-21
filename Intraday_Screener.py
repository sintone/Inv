# -*- coding: utf-8 -*-
"""
@Description: A股盘中实时筛选 (脚本1逻辑: 动态换手率 & 实时价>均线)
@RunTime: 建议在 11:35 和 14:15 运行
"""
import tushare as ts
import pandas as pd
import datetime
import time
import numpy as np
import talib
import requests
import json
import os
import traceback
from tqdm import tqdm

# ================= 配置区域 =================
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK", "")

# 初始化
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
API_CALL_DELAY = 0.02 

# ================= 飞书发送 =================
def send_feishu_msg(title, content):
    if not FEISHU_WEBHOOK_URL:
        print(f"【模拟发送】{title}\n{content}")
        return
    # 获取北京时间用于显示
    beijing_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    current_time = beijing_now.strftime('%m-%d %H:%M')
    
    full_text = f"【{title}】\n{current_time}\n--------------------\n{content}"
    headers = {'Content-Type': 'application/json'}
    payload = {"msg_type": "text", "content": {"text": full_text}}
    try:
        requests.post(FEISHU_WEBHOOK_URL, headers=headers, data=json.dumps(payload), timeout=10)
    except Exception as e:
        print(f"飞书发送报错: {e}")

# ================= 核心工具函数 =================

def get_beijing_now():
    """获取当前的北京时间 (处理GitHub Actions UTC时区问题)"""
    utc_now = datetime.datetime.utcnow()
    return utc_now + datetime.timedelta(hours=8)

def get_last_trade_day_history():
    """获取上一个确定的收盘交易日 (用于获取历史K线)"""
    now = get_beijing_now()
    check_date = now - datetime.timedelta(days=1)
    for _ in range(10):
        date_str = check_date.strftime('%Y%m%d')
        try:
            df = pro.trade_cal(exchange='', start_date=date_str, end_date=date_str)
            if not df.empty and df.iloc[0]['is_open'] == 1:
                return date_str
        except: pass
        check_date -= datetime.timedelta(days=1)
    return None

def calculate_dynamic_threshold(base_threshold=2.5):
    """
    根据当前北京时间，动态计算换手率阈值
    算法: 2.5% * (已交易分钟数 / 全天240分钟)
    """
    now = get_beijing_now()
    
    # 定义关键时间点
    t_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    t_lunch_start = now.replace(hour=11, minute=30, second=0, microsecond=0)
    t_lunch_end = now.replace(hour=13, minute=0, second=0, microsecond=0)
    t_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    minutes_elapsed = 0
    total_minutes = 240.0 # 4小时
    
    if now < t_open:
        minutes_elapsed = 0 # 开盘前
    elif now <= t_lunch_start:
        minutes_elapsed = (now - t_open).total_seconds() / 60
    elif now <= t_lunch_end:
        minutes_elapsed = 120 # 午休期间算作上午结束
    elif now <= t_close:
        morning_minutes = 120
        afternoon_minutes = (now - t_lunch_end).total_seconds() / 60
        minutes_elapsed = morning_minutes + afternoon_minutes
    else:
        minutes_elapsed = 240 # 收盘后
        
    # 计算比例
    ratio = minutes_elapsed / total_minutes
    if ratio > 1.0: ratio = 1.0
    if ratio < 0.05: ratio = 0.05 # 刚开盘给个保底比例，防止除以0或阈值过低
    
    dynamic_val = base_threshold * ratio
    return round(dynamic_val, 2), int(ratio * 100)

def get_realtime_snapshot(stock_basics_df):
    """获取全市场实时行情"""
    print(">> 正在获取全市场实时行情 (Sina接口)...")
    
    # 建立 code (6位) -> ts_code 映射
    code_map = {code.split('.')[0]: code for code in stock_basics_df['ts_code']}
    code_list = list(code_map.keys())
    
    batch_size = 800
    realtime_dfs = []
    
    for i in tqdm(range(0, len(code_list), batch_size), desc="实时数据下载"):
        batch_codes = code_list[i : i + batch_size]
        try:
            df = ts.get_realtime_quotes(batch_codes)
            if df is not None and not df.empty:
                realtime_dfs.append(df)
            time.sleep(0.5) 
        except Exception as e:
            print(f"Batch fetch error: {e}")
            continue
            
    if not realtime_dfs:
        return pd.DataFrame()
    
    full_realtime = pd.concat(realtime_dfs, ignore_index=True)
    
    # 清洗
    full_realtime['price'] = pd.to_numeric(full_realtime['price'], errors='coerce')
    full_realtime['volume'] = pd.to_numeric(full_realtime['volume'], errors='coerce')
    full_realtime = full_realtime[full_realtime['price'] > 0].copy()
    
    full_realtime['ts_code'] = full_realtime['code'].map(code_map)
    
    # 合并基本面 (获取流通股本)
    merged_df = pd.merge(full_realtime, stock_basics_df[['ts_code', 'name', 'float_share', 'industry']], on='ts_code', how='inner')
    
    # 计算实时换手率
    merged_df['turnover_rate_now'] = (merged_df['volume'] / (merged_df['float_share'] * 10000)) * 100
    
    return merged_df

def check_ma_condition_hybrid(ts_code, current_price, history_end_date):
    """混合计算：历史数据 + 当前实时价格 -> 计算 MA"""
    start_date = (pd.to_datetime(history_end_date) - datetime.timedelta(days=300)).strftime('%Y%m%d')
    
    try:
        df_hist = pro.daily(ts_code=ts_code, start_date=start_date, end_date=history_end_date, fields='trade_date,close')
    except:
        return None

    if df_hist is None or df_hist.empty or len(df_hist) < 120:
        return None
    
    # 升序排列
    df_hist = df_hist.sort_values('trade_date', ascending=True)
    
    # 构造新序列
    closes = df_hist['close'].values.tolist()
    closes.append(float(current_price)) 
    close_array = np.array(closes)
    
    # 计算指标
    try:
        ma20 = talib.SMA(close_array, 20)[-1]
        ma60 = talib.SMA(close_array, 60)[-1]
        ma120 = talib.SMA(close_array, 120)[-1]
    except:
        return None
    
    if np.isnan(ma20) or np.isnan(ma60) or np.isnan(ma120):
        return None
        
    # 判断逻辑 (Script 1)
    if current_price > ma20 and current_price > ma60 and current_price > ma120:
        return {
            'ma20': ma20, 'ma60': ma60, 'ma120': ma120
        }
    return None

# ================= 主逻辑 =================

def run_intraday_screener():
    print(">>> 开始执行盘中筛选 (Script 1 - 动态换手率版)...")
    
    # 1. 确定日期
    last_trade_day_hist = get_last_trade_day_history()
    print(f"历史数据基准日: {last_trade_day_hist}")
    if not last_trade_day_hist: return

    # 2. 计算动态阈值
    BASE_THRESHOLD = 2.5
    dynamic_threshold, progress_pct = calculate_dynamic_threshold(BASE_THRESHOLD)
    print(f"当前时间进度: {progress_pct}%")
    print(f"换手率阈值调整: {BASE_THRESHOLD}% -> {dynamic_threshold}%")

    # 3. 获取基础数据
    print("获取基础数据...")
    df_basic = pro.daily_basic(trade_date=last_trade_day_hist, fields='ts_code,float_share')
    df_names = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
    df_base = pd.merge(df_basic, df_names, on='ts_code')
    
    # 4. 获取实时行情
    df_realtime = get_realtime_snapshot(df_base)
    if df_realtime.empty:
        send_feishu_msg("错误", "实时行情获取失败")
        return
    
    # 5. 第一轮筛选：动态换手率
    df_pass_turnover = df_realtime[df_realtime['turnover_rate_now'] > dynamic_threshold].copy()
    print(f"实时换手率 > {dynamic_threshold}% 的股票数: {len(df_pass_turnover)}")
    
    if df_pass_turnover.empty:
        send_feishu_msg("盘中筛选", f"无 (换手率未达标 {dynamic_threshold}%)")
        return

    # 6. 第二轮筛选：混合均线
    final_results = []
    # 限制计算数量，防止超时 (取换手率最高的 300 只进行均线计算)
    df_pass_turnover = df_pass_turnover.sort_values('turnover_rate_now', ascending=False).head(300)
    target_list = df_pass_turnover.to_dict('records')
    
    print("开始计算混合均线...")
    for row in tqdm(target_list):
        ts_code = row['ts_code']
        current_price = float(row['price'])
        
        ma_res = check_ma_condition_hybrid(ts_code, current_price, last_trade_day_hist)
        
        if ma_res:
            row.update(ma_res)
            # 计算实时涨跌幅
            pre_close = float(row['pre_close'])
            if pre_close > 0:
                row['pct_chg_now'] = (current_price - pre_close) / pre_close * 100
            else:
                row['pct_chg_now'] = 0.0
                
            final_results.append(row)
            
    # 7. 生成报告
    df_final = pd.DataFrame(final_results)
    
    msg_lines = []
    # 头部信息增加进度提示
    msg_lines.append(f"时间进度: {progress_pct}% | 动态阈值: >{dynamic_threshold}%")
    
    if not df_final.empty:
        df_final = df_final.sort_values('turnover_rate_now', ascending=False)
        
        msg_lines.append(f"共筛选出 {len(df_final)} 只 (量足价升)")
        msg_lines.append("代码 | 名称 | 行业 | 现价 | 涨幅 | 换手")
        
        # 展示前 20 个
        for _, row in df_final.head(20).iterrows():
            code = row['ts_code'].split('.')[0]
            name = row['name']
            ind = row['industry']
            price = f"{row['price']:.2f}"
            pct = f"{row['pct_chg_now']:.2f}%"
            turn = f"{row['turnover_rate_now']:.2f}%"
            
            msg_lines.append(f"{code}|{name}|{ind}|{price}|{pct}|{turn}")
            
        if len(df_final) > 20:
            msg_lines.append(f"...剩余 {len(df_final)-20} 只见附件")
            
        # 保存 CSV
        timestamp = datetime.datetime.now().strftime('%H%M')
        df_final.to_csv(f'Intraday_{timestamp}.csv', index=False, encoding='utf-8-sig')
        
    else:
        msg_lines.append("换手率达标，但技术形态未确认。")

    send_feishu_msg("A股盘中突围 (Script 1)", "\n".join(msg_lines))
    print("执行完毕。")

if __name__ == "__main__":
    try:
        run_intraday_screener()
    except Exception as e:
        traceback.print_exc()
        send_feishu_msg("盘中脚本出错", str(e))
