# Synthetic Insurance Customer Records

> **Test data only.** Every person, address, policy, payment, and claim in this
> document is fictional. This document must not be used for insurance decisions,
> customer contact, identity verification, or any production workflow.

Data snapshot: 2026-08-30
Currency: CAD

## Customer summary

| Customer ID | Name | Date of birth | Email | Phone | City | Province |
|---|---|---|---|---|---|---|
| SYN-C1001 | Avery North | 1988-04-12 | avery.north@example.com | +1-416-555-0101 | Maple Crossing | ON |
| SYN-C1002 | Jordan Vale | 1976-11-03 | jordan.vale@example.com | +1-647-555-0102 | Cedar Harbour | ON |
| SYN-C1003 | Priya Linden | 1993-07-25 | priya.linden@example.com | +1-604-555-0103 | Rainport | BC |
| SYN-C1004 | Mateo Brooks | 1981-02-17 | mateo.brooks@example.com | +1-403-555-0104 | Prairie Glen | AB |
| SYN-C1005 | Noor Bellamy | 1998-09-09 | noor.bellamy@example.com | +1-514-555-0105 | Rivière-Claire | QC |
| SYN-C1006 | Elise Rowan | 1969-06-30 | elise.rowan@example.com | +1-902-555-0106 | Atlantic View | NS |
| SYN-C1007 | Theo Mercer | 1985-12-14 | theo.mercer@example.com | +1-204-555-0107 | Red Willow | MB |
| SYN-C1008 | Mina Park | 1990-03-21 | mina.park@example.com | +1-306-555-0108 | North Prairie | SK |

## Policies

| Policy ID | Customer ID | Product | Status | Effective date | Renewal date | Annual premium | Coverage summary |
|---|---|---|---|---|---|---:|---|
| SYN-P-A1001 | SYN-C1001 | Auto Plus | Active | 2026-01-15 | 2027-01-15 | $1,680 | $2M liability; collision; comprehensive; $1,000 deductible |
| SYN-P-H1002 | SYN-C1002 | Home Standard | Active | 2025-10-01 | 2026-10-01 | $1,240 | $750K dwelling; $500K liability; $1,500 deductible |
| SYN-P-T1003 | SYN-C1003 | Travel Annual | Active | 2026-05-20 | 2027-05-20 | $460 | Emergency medical up to $5M; 30 days per trip |
| SYN-P-A1004 | SYN-C1004 | Auto Basic | Pending cancellation | 2025-09-08 | 2026-09-08 | $1,410 | $1M liability; comprehensive; $2,000 deductible |
| SYN-P-R1005 | SYN-C1005 | Tenant Plus | Active | 2026-02-01 | 2027-02-01 | $420 | $50K contents; $2M liability; $750 deductible |
| SYN-P-L1006 | SYN-C1006 | Term Life 20 | Active | 2021-06-30 | 2041-06-30 | $1,080 | $500K death benefit; 20-year level term |
| SYN-P-H1007 | SYN-C1007 | Home Plus | Lapsed | 2025-12-12 | 2026-12-12 | $1,560 | $900K dwelling; sewer backup; $1,000 deductible |
| SYN-P-A1008 | SYN-C1008 | Auto Plus | Active | 2026-04-05 | 2027-04-05 | $1,795 | $2M liability; collision; comprehensive; $1,000 deductible |

## Billing and service notes

| Customer ID | Payment method | Account standing | Next payment | Service note |
|---|---|---|---:|---|
| SYN-C1001 | Monthly bank withdrawal | Current | $140.00 on 2026-09-15 | Requested an electronic renewal package. |
| SYN-C1002 | Annual credit card | Current | $1,240.00 on 2026-10-01 | Asked to review the dwelling limit before renewal. |
| SYN-C1003 | Annual credit card | Current | Paid | Added a trip to Iceland for October 2026. |
| SYN-C1004 | Monthly bank withdrawal | Past due | $235.00 overdue | Cancellation is scheduled for 2026-09-08 unless payment clears. |
| SYN-C1005 | Monthly credit card | Current | $35.00 on 2026-09-01 | Updated mailing preference to email only. |
| SYN-C1006 | Annual bank withdrawal | Current | $1,080.00 on 2027-06-30 | Beneficiary review requested; no change recorded yet. |
| SYN-C1007 | Monthly bank withdrawal | Lapsed | $260.00 overdue | Two payments were returned; reinstatement review is required. |
| SYN-C1008 | Monthly credit card | Current | $149.58 on 2026-09-05 | Added winter-tire discount effective 2026-08-20. |

## Claims

| Claim ID | Policy ID | Loss date | Type | Status | Reserve or payment | Adjuster note |
|---|---|---|---|---|---:|---|
| SYN-CL-7001 | SYN-P-A1001 | 2026-07-08 | Rear-end collision | Open | $8,500 reserve | Vehicle inspection completed; repair estimate under review. |
| SYN-CL-7002 | SYN-P-H1002 | 2026-02-19 | Frozen pipe water damage | Closed | $14,260 paid | Repairs completed and final release received. |
| SYN-CL-7003 | SYN-P-T1003 | 2026-06-11 | Delayed baggage | Closed | $620 paid | Receipts validated; electronic payment issued. |
| SYN-CL-7004 | SYN-P-A1004 | 2026-08-03 | Windshield damage | Open | $900 reserve | Glass replacement appointment scheduled. |
| SYN-CL-7005 | SYN-P-R1005 | 2026-05-27 | Bicycle theft | Denied | $0 | Bicycle was not listed under the required high-value endorsement. |
| SYN-CL-7006 | SYN-P-A1008 | 2026-08-22 | Hail damage | Open | $4,200 reserve | Photographs received; appraisal pending. |

## Suggested document-grounding questions

- Which customers have a past-due or lapsed account?
- What is the status and reserve of Avery North's claim?
- Which policy renews next?
- Why was claim `SYN-CL-7005` denied?
- What is the total amount paid on closed claims?
- Which customers currently have open auto claims?
