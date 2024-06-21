import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Candlestick

# 股票名称
stock_name = 'Amazon'
# 从CSV文件读取数据
data = pd.read_csv(f'../datas/{stock_name}.csv')

# 提取需要的列数据
x_data = data['Date'].tolist()
y_data = data[['Open', 'Close', 'Low', 'High']].values.tolist()

# 创建蜡烛图对象并设置数据
candlestick = (
    Candlestick()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(series_name="Stock Prices", y_axis=y_data)
    .set_global_opts(
        title_opts=opts.TitleOpts(title=f'{stock_name} Stock Prices - Chutian Liu'),
        xaxis_opts=opts.AxisOpts(name="Date", type_="category", boundary_gap=True),
        yaxis_opts=opts.AxisOpts(
            name="Price",
            splitline_opts=opts.SplitLineOpts(
                is_show=True, linestyle_opts=opts.LineStyleOpts(width=1)
            )
        ),
        datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        toolbox_opts=opts.ToolboxOpts(is_show=True, feature={"saveAsImage": {}})
    )
)

# 生成HTML文件
candlestick.render("basic_candlestick.html")