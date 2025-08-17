# Scheduling

Scope for the upcoming scheduling utilities in `loto`.

## Scope

This section outlines the goals for `loto.scheduling`, which will model and
solve project schedules under uncertainty. It coordinates tasks, resources and
working calendars to explore possible completion dates.

See the [project objectives](../00_objectives.md) for broader context.

## Glossary

- **Task**: discrete unit of work with an expected duration.
- **Precedence**: dependency indicating that one task must finish before another
  can start.
- **Resource**: limited asset (person, tool, space) assigned to tasks.
- **Calendar**: definition of working time for resources and tasks.
- **Sample**: one simulated realization of a schedule.
- **P50/P80/P90**: percentiles of completion time derived from many samples;
  P50 is the median, P80 the 80th percentile and P90 the 90th percentile.

## High-level Dataflow

1. Define tasks, precedence relations and resource calendars.
2. Generate multiple schedule samples respecting those constraints.
3. Aggregate completion times from samples to estimate P50/P80/P90 milestones.

For details on guidance mechanisms see [Guidance](../03_guidance.md).
