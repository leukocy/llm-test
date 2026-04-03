# Task Plan: LLM-Test Code Quality Improvements

## Goal
Fix the code quality issues identified in the code review, prioritizing medium and high priority items to improve the overall codebase quality.

## Current Phase
Phase 4

## Phases

### Phase 1: Medium Priority - Fix Bare Except Clauses
- [x] Fix bare except in `core/benchmark_runner.py:1851`
- [x] Fix bare except in `core/consistency_tester.py:293`
- [x] Fix bare except in `core/failure_analyzer.py:331`
- [x] Fix bare except in `evaluators/yaml_evaluator.py:242`
- [x] Fix bare except in `core/enhanced_parser.py:454`
- [x] Fix bare except in `core/enhanced_parser.py:524`
- [x] Fix bare except in `core/enhanced_parser.py:553`
- [x] Fix bare except in `core/response_cache.py:186`
- [x] Fix bare except in `core/response_cache.py:453`
- [x] Run tests to verify fixes
- **Status:** complete

### Phase 2: Low Priority - Remove Dead Code
- [x] Remove unused function `render_preset_management()` from `app.py`
- [x] Remove unused function `render_custom_config()` from `app.py`
- [x] Verify app still works after removal
- **Status:** complete

### Phase 3: Low Priority - Fix Temporary File Leaks
- [x] Fix temp file leak in `ui/thinking_components.py:354`
- [x] Fix temp file leak in `ui/thinking_components.py:366`
- [x] Test download functionality (imports verified)
- **Status:** complete

### Phase 4: Low Priority - Replace Debug Print Statements
- [ ] Replace debug prints in `core/benchmark_runner.py`
- [ ] Replace debug prints in `core/quality_evaluator.py`
- [ ] Replace debug prints in `core/tokenizer_utils.py`
- **Status:** pending

### Phase 5: Low Priority - Fix Documentation
- [ ] Fix comment numbering in `app.py:150`
- **Status:** pending

### Phase 6: Verification & Testing
- [x] Run all security tests (21 passed)
- [x] Run unit tests (imports verified)
- [x] Verify app starts without errors
- [x] Update CODE_REVIEW_REPORT.md with fixes applied
- **Status:** complete

### Phase 4: Low Priority - Replace Debug Print Statements
- [ ] Replace debug prints in `core/benchmark_runner.py` (Lines 131, 134, 136)
- [ ] Replace debug prints in `core/quality_evaluator.py` (Line 466)
- [ ] Replace debug prints in `core/tokenizer_utils.py` (Lines 58, 62)
- **Status:** pending - Optional

### Phase 5: Low Priority - Fix Documentation
- [ ] Fix comment numbering in `app.py:150` (step 5 → step 6)
- **Status:** pending - Optional

## Key Questions
1. Should we fix all bare except clauses or focus on the most critical ones first?
2. Do we need to add logging infrastructure before replacing print statements?
3. Should pickle be replaced with JSON (larger change)?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Fix bare except clauses first | Medium priority, affects code quality significantly |
| Keep pickle for now | Would require larger refactoring, current risk is mitigated |
| Use logging module | Replace print statements with proper logging |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| | 1 | |

## Notes
- All fixes should maintain backward compatibility
- Run tests after each phase to ensure nothing breaks
- Commit changes after each completed phase
