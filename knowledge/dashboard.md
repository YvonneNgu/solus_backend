# CloudDash Dashboard Guide

## Overview
Dashboards display your analytics data through customizable widgets. This guide covers widget configuration and troubleshooting.

## Dashboard Types
- **Personal Dashboards**: Your private analytics views
- **Shared Dashboards**: Team collaboration dashboards  
- **Embedded Dashboards**: Can be integrated into other applications

## Widget Types
- **Line Charts**: Time-series data visualization
- **Bar Charts**: Comparison between categories  
- **Tables**: Detailed data in tabular format
- **KPI Cards**: Single metric displays

## Widget Configuration
1. Click the gear icon (⚙️) on any widget
2. Adjust refresh rate (minimum 5 minutes recommended)
3. Configure display options
4. Save changes

## Refresh Rates
- **5 minutes**: Best for real-time monitoring
- **15 minutes**: Good balance of performance and freshness
- **1 hour**: May cause data loading issues

## Common Issues and Solutions

### "Widget Failed to Load" or "No Data" 
- **Cause**: Data source unavailable, permissions issue, or refresh rate too low
- **Step-by-Step Solution**:
  1. **Check widget refresh rate** (most common cause):
     - Click the gear icon (⚙️) on the widget header
     - Look at the current refresh rate setting
     - If set to "1 hour", change to "5 minutes" or "15 minutes"  
     - Click "Update Widget" to apply changes
  2. **Alternative: Check permissions**:
     - Verify you have access to the data source
     - Contact administrator if permission error persists
  3. **Alternative: Refresh dashboard**:
     - Try refreshing the entire dashboard page
     - Check if data source is still available

### "Dashboard Performance Issues"
- **Cause**: Too many widgets or slow refresh rates
- **Solution**: Reduce widgets or increase refresh intervals

### "Time Zone Issues"
- **Cause**: Incorrect time zone settings affecting data display
- **Solution**: Check and adjust dashboard time zone settings

## Best Practices
1. Use appropriate refresh rates (5-15 minutes)
2. Limit widgets per dashboard for better performance
3. Use clear, descriptive widget titles
4. Apply filters to focus on relevant data