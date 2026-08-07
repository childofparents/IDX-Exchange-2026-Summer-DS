# California Single-Family Close Price Prediction

This project evaluates automated valuation model (AVM) candidates for predicting the final `ClosePrice` of a California single-family residence. The final, reproducible experiment uses a chronological, out-of-time holdout: train on June 2025 through May 2026 and test once on June 2026. It is designed to predict value from property and location characteristics, not from seller pricing or sale-process information.

## Project status and headline result

**Week 9 was intentionally skipped.** There is no Streamlit app, `app.py`, or serialized production model in this repository, so there are no app-launch instructions. The final runnable modeling workflow is Week 7; Week 8 contains the saved evaluation results and findings.

No model wins every metric. XGBoost is the recommended statewide candidate because it has the highest holdout R-squared and the lowest RMSE (large-dollar-error control). Random Forest has the lowest MAPE and MdAPE, so it remains the best benchmark for the typical sale's relative error.

| Recommended use | Model | June 2026 holdout evidence |
| --- | --- | --- |
| Primary statewide candidate | XGBoost: depth 8, learning rate 0.05, 500 trees | R-squared **0.812**; RMSE **$666,903**; MAE **$232,373** |
| Typical-relative-error benchmark | Random Forest: 60 trees, depth 30, leaf size 10, max features 0.7 | MAPE **23.69%**; MdAPE **9.14%** |

## Dataset source and snapshot

The source is licensed historical sold-listing data from **CRMLS (California Regional Multiple Listing Service)**. Local monthly extracts are stored as `raw data/CRMLSSoldYYYYMM.csv`; they are source snapshots and must not be edited. Obtain any replacement extracts through the team's authorized data-access process rather than adding connection credentials to this repository.

`ClosePrice` is the target: the amount paid by the buyer to the seller when the transaction closes. The final experiment uses only records satisfying all of the following:

- `StateOrProvince = CA`
- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`
- `ClosePrice > 0`

| Snapshot/checkpoint | Rows | Notes |
| --- | ---: | --- |
| Raw files imported | 283,176 | 13 consecutive files: `CRMLSSold202506.csv` through `CRMLSSold202606.csv` |
| Eligible CA single-family sales | 143,071 | Applied geography, property-type, subtype, and positive-target filter |
| Training rows before target handling | 130,214 | June 2025-May 2026 |
| Training target outliers removed | 154 | Determined using training data only |
| Final training rows | 130,060 | Used for final fitting |
| Untouched test rows | 12,857 | Every eligible June 2026 sale; **0 rows removed** |

The script verifies that each row's `CloseDate` month agrees with its source-file month. The recorded final run found zero month mismatches.

Primary references are the [project brief](Data%20Science%20v.4.pdf), [real-estate primer](Real_Estate_Primer.pdf), [Trestle Property metadata](resources/Trestle%20Property%20MetaData.pdf), and [Week 1 key-column notes](Week1_Key_Column_Notes.txt).

## Preprocessing and feature engineering

The final [Week 7 script](W7%20Advanced%20Models/05_advanced_models.py) recreates the Week 3 cleaning and Week 6 feature engineering from the raw monthly data. That makes the final reported result reproducible without relying on a notebook's in-memory state.

1. **Scope and schema checks.** The script requires 13 contiguous monthly files, validates required columns, parses dates/numerics, and applies the eligible-record filter above.
2. **Invalid values become missing, not dropped rows.** Implausible coordinates, non-positive living area/lot size/bedroom/bathroom counts, negative parking/garage/association fees, inconsistent main-level bedrooms, and extreme count values are converted to missing values. This preserves an otherwise valid sale for later imputation.
3. **Text and categorical cleanup.** Postal codes are reduced to a five-digit value; blank and ambiguous school labels become missing; multi-valued `Flooring` and `Levels` are converted to indicator features; source missingness is retained with indicators.
4. **Training-only target handling and caps.** The script removes 154 train-only target outliers using the lower 0.1% `ClosePrice` threshold and an extreme price-per-living-area rule among the highest-priced training sales. It does not trim June 2026. The 99.9th-percentile training caps for `LivingArea`, `LotSizeSquareFeet`, and `AssociationFee` are then applied to both splits.
5. **Fold-fitted transformations.** Numeric values use median imputation with missing indicators and standardization. Categorical values use an `Unknown` imputation value and one-hot encoding; categories occurring fewer than 50 times are handled safely. These transformations are fit within each validation fold, then refit on the full training period. The existing final run uses the reported `AssociationFee` amount without `AssociationFeeFrequency` normalization; that is a documented limitation to correct in a future run.
6. **Engineered inputs.** The model adds `BedBathRatio`, `PropertyAgeAtSale`, cyclical month features, log living/lot area, and a CDE 2025-26 unified-school-district point-in-polygon feature plus a boundary-match flag. The final transformed matrix contains 1,193 encoded features.

### Leakage and availability controls

`ClosePrice` is target-only. `ListPrice`, `OriginalListPrice`, `DaysOnMarket`, `PurchaseContractDate`, `ListingContractDate`, `ContractStatusChangeDate`, `PricePerLivingArea`, identifiers, agent/office data, and unparsed addresses are excluded from the final model matrix. Listing-process fields are either too close to the outcome or unavailable at a true valuation query.

`CloseDate` and source-month fields are trace/split fields, not direct inputs. `CloseDate` is used to derive the documented seasonal features in this historical experiment. A production valuation workflow should supply the query/valuation month instead of relying on a future closing date.

## Models tested and validation design

The project began with Linear Regression, then tested Decision Tree and Random Forest models, added engineered features, and finally tested gradient boosters (XGBoost and LightGBM). Fixed seed: **55**. The final tuning uses complete-month chronological validation within the training period (March-May 2026); June 2026 was held aside until model selection was complete. Earlier Week 4-6 results use a May 2026 holdout and are historical context only, not an apples-to-apples comparison with the final table.

| Model | Selected configuration | R-squared | MAE | RMSE | MAPE | MdAPE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | Linear baseline | 0.699 | $320,643 | $843,525 | 33.82% | 17.98% |
| Decision Tree | depth 24; min samples/leaf 10 | 0.722 | $270,596 | $810,674 | 26.30% | 10.61% |
| Random Forest | 60 trees; depth 30; leaf 10; max features 0.7 | 0.749 | $232,646 | $770,385 | **23.69%** | **9.14%** |
| XGBoost | depth 8; learning rate 0.05; 500 trees | **0.812** | **$232,373** | **$666,903** | 25.20% | 11.77% |
| LightGBM | depth 10; learning rate 0.05; 500 trees; 63 leaves | 0.798 | $242,444 | $691,360 | 26.56% | 12.23% |

Lower is better for MAE, RMSE, MAPE, and MdAPE; higher is better for R-squared. R-squared measures explained variation, MAE/RMSE measure dollar error (with RMSE emphasizing large misses), and MAPE/MdAPE measure relative error. MdAPE is the preferred typical-error headline because it is scale-invariant and robust to the right-skewed price-error tail; it should be reported alongside RMSE and MAE rather than replacing them.

Price-band findings reinforce the tradeoff: XGBoost performs best around $800k-$1.1M (9.88% MdAPE) and is weakest below $575k (15.06% MdAPE). Random Forest is strongest at $575k-$800k (6.24% MdAPE) and weakest at $1.67M+ (14.10% MdAPE). See [Week 8 findings](W8%20Evaluation%20Expansion/Week8_Evaluation_Findings.docx) and [metrics summary](W8%20Evaluation%20Expansion/metrics_summary.csv) for the full evaluation.

## Re-run instructions

### Reproduce the final reported model comparison

Run these commands from the repository root. Python 3.13 is the recorded tested environment for Week 8 and a compatible choice for the Week 7 dependencies. The commands create an isolated environment; do not install packages into the system interpreter.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r "W7 Advanced Models/requirements.txt"
python "W7 Advanced Models/05_advanced_models.py"
```

Before running, confirm that these inputs are present and unchanged:

```text
raw data/CRMLSSold202506.csv ... raw data/CRMLSSold202606.csv
W6 Feature Engineering/resources/California_School_District_Areas_2025-26.geojson
```

The script rewrites the Week 7 metric, validation, prediction, feature-importance, and report artifacts in `W7 Advanced Models/`. It finds the project root automatically when started from the repository root; if running from elsewhere, set `IDX_PROJECT_ROOT` to the absolute project path. Review `advanced_model_test_metrics.csv` after completion. The script uses fixed seeds, but CPU library/platform changes can still cause small numerical differences.

### Re-run the earlier weekly artifacts

The Week 4-6 scripts reproduce their own historical May 2026 experiments, using the retained Week 3 `Kelvin` modeling artifacts. They are useful for tracing the model-development sequence, but they are not a prerequisite for the self-contained Week 7 rerun above.

```bash
source .venv/bin/activate
python -m pip install -r "W7 Advanced Models/requirements.txt"
python "W3 Data Preprocessing/02_preprocessing.py"
python "W4 Baseline Model/03_baseline_model.py"
python "W5 Additional Models/04_model_comparison.py"
python "W6 Feature Engineering/05_feature_engineering.py"
```

Important handoff note: `02_preprocessing.py` creates `W3 Data Preprocessing/Cleaned SFR CRMLSSold CSVs/`, whereas the historical Week 4-6 scripts read the versioned `W3 Data Preprocessing/Kelvin/` tables. Do not claim that the first command regenerates those `Kelvin` artifacts; retain them to reproduce the historical outputs. Week 7 reads the raw files directly and is the authoritative rerun path for the final result.

Week 8 has saved evaluation tables and a written findings document, but no executable evaluation script is currently included. Regenerating its exact tables from a fresh Week 7 run requires implementing that missing evaluation step; this is a known reproducibility gap, not an implied automated command.

## Outputs and repository map

| Location | Purpose |
| --- | --- |
| `raw data/` | Immutable local CRMLS sold-listing source extracts |
| `W3 Data Preprocessing/02_preprocessing.py` | Source validation, filtering, and cleaned-data generation |
| `W4 Baseline Model/03_baseline_model.py` | Linear Regression baseline and May 2026 historical results |
| `W5 Additional Models/04_model_comparison.py` | Decision Tree and Random Forest comparison |
| `W6 Feature Engineering/05_feature_engineering.py` | Engineered features and school-district spatial join |
| `W7 Advanced Models/05_advanced_models.py` | Final reproducible XGBoost/LightGBM workflow and June 2026 evaluation |
| `W7 Advanced Models/advanced_model_test_metrics.csv` | Final Week 7 three-model holdout metrics |
| `W8 Evaluation Expansion/metrics_summary.csv` | Same-month five-model summary used in the results table |

## Known limitations and next steps

- The evidence is one out-of-time month, not a multi-month rolling backtest. Re-evaluate monthly and monitor error by county/ZIP and price band before production selection.
- Entry-level and luxury properties have weaker relative-error behavior; statewide metrics can hide local-market underperformance.
- The historical `CloseDate` seasonal feature must be replaced with a known valuation/query date in a live prediction workflow.
- The data is limited to California residential single-family sales and can drift as inventory, interest rates, and reporting practices change.
- `AssociationFee` is not normalized by `AssociationFeeFrequency` in the final run, so fee amounts on different schedules are not directly comparable.
- Week 9 was skipped; a production app still needs a saved preprocessing pipeline, model artifact, input-schema assertion, monitoring, and a prediction-interval/uncertainty design.
- Add a versioned Week 8 evaluation script and a data manifest/checksum so the complete reporting chain can be rebuilt from scratch.

## Key-column metadata notes

The tables below follow the Trestle `Field | Type | Size | Description` convention, with an added project-use column. They intentionally include **only** columns named in [Week 1 key-column notes](Week1_Key_Column_Notes.txt), not the full CRMLS extract or the full Trestle Property resource. Sizes and native types come from the Trestle metadata; project treatment records how each field should be used in this AVM.

### DateTime

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `ListingContractDate` | DateTime | - | Effective date of the seller-broker agreement. | Exclude from final model; sale-process timing. |
| `PurchaseContractDate` | DateTime | - | Date an offer was accepted. | Exclude from final model; outcome-adjacent. |
| `CloseDate` | DateTime | - | Date the purchase agreement was fulfilled. | Split/audit field; derive seasonal feature only in historical workflow. |
| `ContractStatusChangeDate` | DateTime | - | Contractual listing-status-change date. | Exclude from final model; sale-process timing. |

### Decimal

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `ClosePrice` | Decimal | 14.2 | Amount paid by purchaser to seller. | Target only; never a feature. |
| `ListPrice` | Decimal | 14.2 | Current asking price set by seller/broker. | Exclude; pricing leakage. |
| `OriginalListPrice` | Decimal | 14.2 | Price in the initial listing agreement. | Exclude; pricing leakage. |
| `TaxAnnualAmount` | Decimal | 14.2 | Annual property tax from latest assessment. | Candidate only after missingness/jurisdiction checks. |
| `AssociationFee` | Decimal | 14.2 | HOA fee for common-area/neighborhood benefits. | Clean and cap; current final run does not normalize fee frequency. |
| `LivingArea` | Decimal | 14.2 | Total livable area within the structure. | Core feature; clean and cap; also log-transform. |
| `BuildingAreaTotal` | Decimal | 14.2 | Total structure area, finished and unfinished. | Candidate; check overlap with `LivingArea`. |
| `AboveGradeFinishedArea` | Decimal | 14.2 | Finished area at/above ground surface. | Candidate; assess sparsity. |
| `BelowGradeFinishedArea` | Decimal | 14.2 | Finished area below ground surface. | Candidate; assess sparsity. |
| `LotSizeSquareFeet` | Decimal | 14.2 | Total lot square footage. | Core feature; clean, cap, and log-transform. |
| `LotSizeAcres` | Decimal | 16.4 | Total lot acreage. | Alternative lot-size representation; reconcile with square feet. |
| `LotSizeArea` | Decimal | 16.4 | Total lot area; units are external context. | Candidate only after unit validation. |
| `Latitude` | Decimal | 12.8 | Latitude of a property reference point. | Clean California range; supports spatial features. |
| `Longitude` | Decimal | 12.8 | Longitude of a property reference point. | Clean California range; supports spatial features. |
| `ParkingTotal` | Decimal | 14.2 | Total parking spaces included in sale. | Candidate after non-negative/range checks. |
| `GarageSpaces` | Decimal | 14.2 | Number of garage spaces. | Candidate after non-negative/range checks. |
| `CoveredSpaces` | Decimal | 14.2 | Total garage and carport spaces. | Candidate; check redundancy with parking/garage counts. |

### Integer (`Int32`/`Int64`)

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `ListingKeyNumeric` | Int64 | 10 | Numeric listing-key representation. | Identifier only; exclude. |
| `DaysOnMarket` | Int32 | 4 | Days on market under MLS business rules. | Exclude from final model; availability/leakage concern. |
| `TaxYear` | Int32 | 4 | Year of the tax assessment. | Context for `TaxAnnualAmount`; candidate feature. |
| `BedroomsTotal` | Int32 | 4 | Total bedrooms in the dwelling. | Core feature; clean invalid counts. |
| `BathroomsTotalInteger` | Int32 | 4 | Integer sum of bathrooms. | Core feature; clean invalid counts and build bed/bath ratio. |
| `MainLevelBedrooms` | Int32 | 4 | Bedrooms on the main/entry level. | Candidate; validate against total bedrooms. |
| `Stories` | Int32 | 4 | Floors in the property. | Candidate feature. |
| `YearBuilt` | Int32 | 4 | Year of initial habitability. | Candidate; derive age at sale. |
| `StreetNumberNumeric` | Int32 | 4 | Integer portion of street number. | Address identifier; exclude. |
| `FireplacesTotal` | Int32 | 4 | Total fireplaces. | Candidate feature. |

### Boolean

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `NewConstructionYN` | Boolean | - | Newly constructed and not previously occupied. | Candidate flag; normalize values and retain missingness. |
| `AttachedGarageYN` | Boolean | - | At least one garage is attached to primary dwelling. | Candidate flag. |
| `FireplaceYN` | Boolean | - | Property includes a fireplace. | Candidate flag. |
| `PoolPrivateYN` | Boolean | - | Property has a private pool included in sale. | Candidate flag. |
| `ViewYN` | Boolean | - | Property has a view. | Candidate flag. |
| `WaterfrontYN` | Boolean | - | Property is on waterfront. | Candidate flag. |
| `BasementYN` | Boolean | - | Property has a basement. | Candidate flag; assess local prevalence. |

### Enumerated/categorical types

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `MlsStatus` | MlsStatus Enum | - | MLS-local listing status. | Audit/filter context; do not rely on as price feature. |
| `AssociationFeeFrequency` | FeeFrequency Enum | - | Frequency an association fee is paid. | Needed to normalize/interpret `AssociationFee`. |
| `PropertyType` | PropertyType Enum | - | Broad property category. | Required filter: `Residential`. |
| `PropertySubType` | PropertySubType Enum | - | Specific property subtype. | Required filter: `SingleFamilyResidence`. |
| `BusinessType` | BusinessType Enum | - | Type of business being sold. | Out of scope for single-family model. |
| `Levels` | Levels Enum | - | Number/layout of property levels. | Parse multi-value entries into indicators. |
| `StateOrProvince` | StateOrProvince Enum | - | Postal abbreviation for state/province. | Required final-experiment filter: `CA`. |
| `Flooring` | Flooring Enum | - | Flooring types in the property. | Parse multi-value entries into material indicators. |
| `ListAgentAOR` | AOR Enum | - | Listing agent's REALTOR association/board. | Participant field; exclude. |
| `BuyerAgentAOR` | AOR Enum | - | Buyer's agent's REALTOR association/board. | Post-transaction participant field; exclude. |
| `BuyerOfficeAOR` | AOR Enum | - | Buyer's office REALTOR association/board. | Post-transaction participant field; exclude. |

### String

| Field | Trestle type | Size | Description | Project treatment |
| --- | --- | ---: | --- | --- |
| `ListingKey` | String | 20 | Primary unique identifier for the Property record. | Deduplication/audit only; exclude. |
| `ListingId` | String | 255 | Human-facing listing identifier. | Retrieval/audit only; exclude. |
| `BuyerAgentMlsId` | String | 25 | Local identifier for buyer's agent. | High-cardinality participant ID; exclude. |
| `BuilderName` | String | 50 | Builder or builder-tract name. | High-cardinality; clean/encode only if justified. |
| `LotSizeDimensions` | String | 150 | Text dimensions of the lot. | Manual review; parse only with a documented method. |
| `UnparsedAddress` | String | 255 | Full civic address as one text value. | Review/geocoding only; exclude raw text. |
| `City` | String | 50 | City in listing address. | Geographic categorical candidate. |
| `PostalCode` | String | 10 | Postal-code portion of address. | Clean to five digits; categorical/location feature. |
| `CountyOrParish` | String | 50 | County or other regional authority. | Geographic categorical candidate. |
| `MLSAreaMajor` | String | 50 | Major MLS-defined marketing area. | Geographic categorical candidate if populated. |
| `SubdivisionName` | String | 50 | Neighborhood/community/tract. | High-cardinality; clean/encode only if justified. |
| `ElementarySchool` | String | 50 | Associated elementary school. | School/location candidate; check cardinality. |
| `MiddleOrJuniorSchool` | String | 50 | Associated middle/junior school. | School/location candidate; check cardinality. |
| `HighSchool` | String | 50 | Associated high school. | School/location candidate; check cardinality. |
| `ElementarySchoolDistrict` | String | 50 | Associated elementary district. | School/location candidate; check cardinality. |
| `MiddleOrJuniorSchoolDistrict` | String | 50 | Associated middle/junior district. | School/location candidate; check cardinality. |
| `HighSchoolDistrict` | String | 50 | Associated high-school district. | Clean ambiguous values; categorical/location input. |
| `ListAgentEmail` | String | 80 | Listing agent email address. | Contact identifier; exclude. |
| `ListAgentFirstName` | String | 50 | Listing agent first name. | Participant identifier; exclude. |
| `ListAgentLastName` | String | 50 | Listing agent last name. | Participant identifier; exclude. |
| `ListAgentFullName` | String | 150 | Full listing-agent name. | Participant identifier; exclude. |
| `ListOfficeName` | String | 255 | Listing brokerage legal name. | Participant field; exclude. |
| `BuyerAgentFirstName` | String | 50 | Buyer's agent first name. | Post-transaction participant field; exclude. |
| `BuyerAgentLastName` | String | 50 | Buyer's agent last name. | Post-transaction participant field; exclude. |
| `BuyerOfficeName` | String | 255 | Buyer's brokerage legal name. | Post-transaction participant field; exclude. |
| `CoListAgentFirstName` | String | 50 | Co-listing agent first name. | Participant identifier; exclude. |
| `CoListAgentLastName` | String | 50 | Co-listing agent last name. | Participant identifier; exclude. |
| `CoListOfficeName` | String | 255 | Co-listing brokerage legal name. | Participant field; exclude. |
| `CoBuyerAgentFirstName` | String | 50 | Buyer's co-agent first name. | Post-transaction participant field; exclude. |

## Documentation provenance

This README records the source snapshot, row counts, feature logic, leakage exclusions, selected hyperparameters, validation design, metrics, limitations, and next steps so a teammate can reproduce or extend the work without relying on undocumented assumptions. It follows the documentation and reproducibility expectations on pages 7-8 of [AVM Data Science Best Practices](AVM_Data_Science_Best_Practices%20v.1.pdf).
