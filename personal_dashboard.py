import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load and prepare the cleaned data
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=50, freq='D'),
    'Category': ['Food', 'Transport', 'Shopping', 'Health', 'Others'] * 10,
    'Description': ['Example'] * 50,
    'Amount': [100, 200, 150, 250, 300] * 10,
    'Type': ['Expense', 'Income'] * 25
})
df['Date'] = pd.to_datetime(df['Date'])

# Filtered copy
df_cleaned = df[['Date', 'Category', 'Description', 'Amount', 'Type']].copy()

# Initialize the Dash app
app = dash.Dash(__name__)

# External CSS for styling
app.css.append_css({"external_url": "/static/styles.css"})

# Layout of the Dash app with added filters
app.layout = html.Div([
    html.H1("Personal Financial Dashboard", className="header-title"),

    dcc.Dropdown(
        id='time-range',
        options=[
            {'label': 'Daily', 'value': 'Daily'},
            {'label': 'Weekly', 'value': 'Weekly'},
            {'label': 'Monthly', 'value': 'Monthly'},
            {'label': 'Yearly', 'value': 'Yearly'},
            {'label': 'Custom Dates', 'value': 'Custom'}
        ],
        value='Monthly',
        clearable=False,
        className="dropdown"
    ),

    dcc.DatePickerRange(
        id='custom-date-picker',
        start_date=df_cleaned['Date'].min(),
        end_date=df_cleaned['Date'].max(),
        style={'display': 'none'}
    ),

    dcc.Dropdown(
        id='transaction-type',
        options=[
            {'label': 'All', 'value': 'All'},
            {'label': 'Income', 'value': 'Income'},
            {'label': 'Expense', 'value': 'Expense'}
        ],
        value='All',
        clearable=False,
        className="dropdown"
    ),

    dcc.Graph(id='transactions-time-series', className="graph-container"),
    dcc.Graph(id='category-pie-chart', className="graph-container"),
    dcc.Graph(id='monthly-spending-bar-chart', className="graph-container")
])

@app.callback(
    Output('custom-date-picker', 'style'),
    [Input('time-range', 'value')]
)
def show_date_picker(time_range):
    return {'display': 'block'} if time_range == 'Custom' else {'display': 'none'}

@app.callback(
    [Output('transactions-time-series', 'figure'),
     Output('category-pie-chart', 'figure'),
     Output('monthly-spending-bar-chart', 'figure')],
    [Input('time-range', 'value'),
     Input('transaction-type', 'value'),
     Input('custom-date-picker', 'start_date'),
     Input('custom-date-picker', 'end_date')]
)
def update_graphs(time_range, transaction_type, start_date, end_date):
    filtered_df = df_cleaned[df_cleaned['Type'] == transaction_type] if transaction_type != 'All' else df_cleaned.copy()

    if time_range == 'Custom':
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) & (filtered_df['Date'] <= pd.to_datetime(end_date))]
    elif time_range == 'Daily':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.date).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Weekly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Monthly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('M').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()
    elif time_range == 'Yearly':
        filtered_df = filtered_df.groupby(filtered_df['Date'].dt.to_period('Y').apply(lambda r: r.start_time)).agg({'Amount': 'sum'}).reset_index()

    if filtered_df.empty:
        return {}, {}, {}

    time_series_fig = px.line(filtered_df, x='Date', y='Amount', title='Transactions Over Time')

    pie_df = df_cleaned[(df_cleaned['Date'] >= pd.to_datetime(start_date)) & (df_cleaned['Date'] <= pd.to_datetime(end_date))] if time_range == 'Custom' else df_cleaned[df_cleaned['Date'].isin(filtered_df['Date'])]
    pie_df = pie_df[pie_df['Type'] == transaction_type] if transaction_type != 'All' else pie_df
    category_pie_chart = px.pie(pie_df, names='Category', values='Amount', title='Category-wise Distribution')

    monthly_spending_fig = px.bar(filtered_df, x='Date', y='Amount', title='Spending Over Selected Time Range')

    return time_series_fig, category_pie_chart, monthly_spending_fig

if __name__ == '__main__':
    app.run_server(debug=True)
