---
type: metric
title: "Gross Margin"
description: "Percentage of total sales revenue remaining after deducting direct cost of goods sold (COGS)."
resource: "looker://models/finance/gross_margin"
tags: ["finance", "kpi", "executive"]
owner: "finance-analytics"
status: "active"
generated:
  by: "finance-governance-bot"
  at: "2026-05-15T10:00:00Z"
  verified_by: "corporate-controller"
  confidence: 1.0
---

# Gross Margin Definition

Gross Margin assesses financial efficiency across all retail categories.

### Formula
$$\text{Gross Margin} = \frac{\text{Net Revenue} - \text{COGS}}{\text{Net Revenue}} \times 100$$

### Upstream Dependencies
- Source transaction records: [Customer Orders](/tables/orders.md)
- Filter condition: Exclude rows where `status` is `CANCELLED` or `REFUNDED` according to [Refund SLA Policy](/policies/refund_sla.md).
