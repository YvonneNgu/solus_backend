# CloudDash Permissions Guide

## Overview
CloudDash uses role-based access control to manage user permissions. This guide covers user roles and common permission issues.

## Permission Levels

### User Roles

| Role | Description | Capabilities |
|------|-------------|--------------|
| **Viewer** | Basic access role | • View assigned dashboards<br>• Export data from viewable dashboards<br>• Save personal dashboard layouts |
| **Analyst** | Data analysis role | • All Viewer capabilities<br>• Create personal dashboards<br>• Create and modify widgets<br>• Access to raw data exports |
| **Manager** | Team leadership role | • All Analyst capabilities<br>• Create and share team dashboards<br>• Manage team members' access<br>• Configure alerts and schedules |
| **Admin** | System administration | • All Manager capabilities<br>• User management<br>• System configuration<br>• Security settings<br>• Billing management |

### Permission Scope
- **Organization Level**: Global settings and user management
- **Dashboard Level**: Access to specific dashboards and widgets
- **Data Source Level**: Access to underlying data

## Permission Inheritance
Permissions flow from organization level down to individual resources. More specific permissions override general ones.

## Common Issues and Solutions

### "Access Denied" Errors
- **Cause**: User lacks necessary permissions for the resource
- **Solution**:
  1. Check user's role in the current workspace
  2. Verify if resource has restricted permissions
  3. Request access from resource owner or admin

### "Missing Dashboards"
- **Cause**: Dashboards are not shared with the user
- **Solution**:
  1. Verify dashboard exists (admin can check)
  2. Check if dashboard is in a different workspace
  3. Request sharing permissions from dashboard owner

### "Cannot Edit Widget"
- **Cause**: User has Viewer permissions on a dashboard
- **Solution**:
  1. Request edit permissions from dashboard owner
  2. Create a personal copy of the dashboard (if allowed)
  3. Check if dashboard is locked for editing

## Managing Permissions

### Adding Users
1. Navigate to Admin Panel
2. Go to User Management
3. Click "Add User" and enter details
4. Assign appropriate role
5. Send invitation

### Permission Audit
1. Navigate to Admin → Security
2. Use "Permission Audit" to review access
3. Make adjustments as needed

## Best Practices
1. Use principle of least privilege
2. Regular permission audits
3. Use groups for easier management
4. Document access for critical resources