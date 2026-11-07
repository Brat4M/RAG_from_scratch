---
type: table
title: "Customer Orders"
description: "One row per completed customer transaction across web, mobile, and in-store channels."
resource: "bigquery://prod-data-platform/sales/orders"
tags: ["sales", "orders", "revenue", "tier-1"]
owner: "sales-engineering"
status: "active"
generated:
  by: "data-lineage-agent/v2"
  at: "2026-05-10T08:30:00Z"
  verified_by: "head-of-data"
  confidence: 1.0
---

# Customer Orders Schema

The `orders` table records immutable records of customer financial commitments.

| Column | Type | Description |
|---|---|---|
| `order_id` | STRING | Primary key (UUID v4) |
| `customer_id` | STRING | Foreign key to [Customer Profiles](/tables/customers.md) |
| `gross_amount` | NUMERIC | Total order amount in USD before discounts |
| `net_amount` | NUMERIC | Settled order amount in USD after merchant discounts |
| `status` | STRING | Transaction status (`COMPLETED`, `REFUNDED`, `CANCELLED`) |
| `created_at` | TIMESTAMP | ISO 8601 creation time |

### Cross References
- Foreign entity: [Customer Profiles](/tables/customers.md)
- Metric consumers: [Gross Margin](/metrics/gross_margin.md)
