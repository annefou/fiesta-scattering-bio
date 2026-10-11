# Published nanopub chain — URI registry

This file is the canonical registry of published nanopub URIs for this replication.
The chain was published before this ledger existed; the URIs below are copied from the
chain table in `README.md`.

## Chain

| Step | Template | URI | Published |
|---|---|---|---|
| 01 | Quote-with-comment | https://w3id.org/sciencelive/np/RAH71F0FaKHQ2OZ-k_mGaO0GB501kDHo_ZqY6ScBbtep4 | before this ledger |
| 02 | AIDA Sentence | https://w3id.org/np/RAT-PGqwhe4Y2hALiOhgEm4ErYVK9vfk0YbQpV2qdN1Ck | before this ledger |
| 03 | FORRT Claim | https://w3id.org/sciencelive/np/RAeI2cspabQBUMHiA94jOKjiBypsIFSPPDwH_f5l1sJu4 | before this ledger |
| 04 | FORRT Replication Study | https://w3id.org/sciencelive/np/RAauQR3eY4NGILF2ei5a6YLbZjT7JDK_tQGYyl2R4DcMk | before this ledger |
| 05 | FORRT Replication Outcome | https://w3id.org/sciencelive/np/RAIqyPwP8PzMKSKCMKb04vzR1CpttIsAYqIRy6RkaBPZk | before this ledger |
| 06 | CiTO Citation | https://w3id.org/sciencelive/np/RA8BSfq4Cbs3A4chU6fvafk0px-yDZwsYKFIO9cnzkyx4 | before this ledger |

Step 02 is on `https://w3id.org/np/…` rather than `…/sciencelive/np/…`: it was published via
Nanodash because the Science Live AIDA form failed when both *datasets* and *publications*
were filled (see README). Both namespaces are valid and citable.

## Verification

`verify_chain` (forrt-research MCP) run on 2026-10-08 against both the production and the
dev Science Live API: all steps reachable, the Outcome's archived version DOI resolves, and the
cited DOI resolves — green.
