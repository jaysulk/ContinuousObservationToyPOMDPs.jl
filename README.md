# ContinuousObservationToyPOMDPs.jl

## Simple Light-Dark

[https://slides.com/zacharysunberg/defense-4#/39](https://slides.com/zacharysunberg/defense-4#/39)

## Continuous Observation Tiger

**S** = {left, right} (where the tiger is)

**A** = {left, right, listen1, listen2} (open a door or listen)

**O** = [0, 1]

**R**:
- -100 if the door with the tiger is opened,
- +10 if the other door is opened
- -1 for listen1
- -2 for listen2

**T**: static state, problem ends when a door is opened

**Z**: [0, 0.5] means tiger in left, [0.5, 1.0] is tiger in right. If action is listen1, 60% chance of observation in correct range; if listen2, 90% chance of observation in correct range

## Usage

See tests; feel free to reach out for more documentation.
