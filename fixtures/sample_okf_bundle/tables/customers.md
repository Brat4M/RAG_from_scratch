---
type: table
title: "Customer Profiles"
description: "Master dimensional table for verified corporate and consumer accounts."
resource: "bigquery://prod-data-platform/crm/customers"
tags: ["crm", "customer", "identity"]
owner: "crm-ops"
status: "active"
generated:
  by: "crm-sync-worker/1.4"
  at: "2026-05-10T09:00:00Z"
  verified_by: "privacy-officer"
  confidence: 0.98
---

# Customer Profiles Schema

Master customer account properties and risk segmentation.

| Column | Type | Description |
|---|---|---|
| `customer_id` | STRING | Primary key |
| `email` | STRING | Primary hashed contact email (PII masked) |
| `tier` | STRING | Loyalty level (`STANDARD`, `GOLD`, `PLATINUM`) |
| `dispute_risk_score` | FLOAT | Automated fraud probability score [0.0 - 1.0] |

### Connected Concepts
- Transactions: [Customer Orders](/tables/orders.md)
- Subject to governance: [Refund SLA Policy](/policies/refund_sla.md)
