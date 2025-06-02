# CloudDash Data Export Guide

## Overview
Export your CloudDash data in multiple formats for analysis and reporting. This guide covers export formats, limitations, and troubleshooting.

## Export Formats

### CSV Format
- **Best for**: Simple tabular data and spreadsheet imports
- **Limitations**: Maximum 10,000 rows
- **Use case**: Monthly reports, simple metrics tracking
- **Common error**: "Export size exceeded" - This occurs when trying to export more than 10,000 rows in CSV format

### XLSX Format
- **Best for**: Larger datasets, preserving formatting
- **Limitations**: Maximum 100,000 rows
- **Use case**: Quarterly reports, financial analysis
- **Required for**: Any dataset exceeding 10,000 rows

### JSON Format
- **Best for**: Preserving data hierarchies and structure
- **Limitations**: More complex to work with in spreadsheet applications
- **Use case**: System integrations, API connections
- **Special note**: Includes full metadata unlike other formats

## Export Process
1. Navigate to **Data Export** page
2. Select your dataset
3. Choose export format (CSV, XLSX, or JSON)
4. Click "Generate Export"
5. Download when ready

## Permissions Required

| Export Type | Required Permission |
|-------------|---------------------|
| Dashboard | Viewer access or higher |
| Raw data | Analyst access or higher |
| User data | Admin access only |
| System logs | Admin access only |

## Common Issues and Solutions

### "Export Failed: Size Exceeded"
- **Cause**: Trying to export too many rows in a format with row limitations
- **Step-by-Step Solution**: 
  1. **Switch to XLSX format** (recommended for datasets between 10,000 and 100,000 rows):
     - Look for the Format section on the export form
     - Select the "XLSX" radio button instead of "CSV"
     - Click "Generate Export" again
  2. **Alternative: Apply filters** to reduce the dataset size:
     - Use the Date Range fields to select a smaller time period
     - This will reduce the number of rows in your export
  3. **Alternative: Break into smaller exports**:
     - Export data in smaller date ranges (e.g., monthly instead of yearly)
     - Combine the files afterward if needed

### "Export Failed: Permission Denied"
- **Cause**: User doesn't have sufficient permissions for the requested data
- **Solution**:
  1. Request elevated permissions from your administrator
  2. Export only the datasets you have permission to access

### "Export Timed Out"
- **Cause**: Export request too large or complex
- **Solution**: Reduce dataset size with filters or break into smaller exports

## Best Practices
1. Use XLSX for larger datasets (10,000+ rows)
2. Apply filters to export only needed data
3. Use JSON for system integrations
4. Schedule large exports during off-peak hours