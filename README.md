# coarse_targeting

A small simulation of **coarse targeting in social networks**: you have one message,
you cannot personalise it, and the only thing you choose is how much exposure each
person gets. Who should you aim at?

Built as a teaching companion to Wei Li and Philip Solimine, *Targeting persistent
disagreement: coarse interventions in social networks* (ACM EC 2026).

> This is a simplified teaching version of the model, not a reimplementation of the
> paper. The paper solves the planner's problem analytically with a value function
> and a Riccati matrix; here the optimum is found numerically on small networks.

## The model

Opinions live in [0, 1] and update by DeGroot averaging plus a broadcast message:

```
x(t+1) = (I - B) A x(t) + B m
```

- `A` is the row-stochastic listening matrix (who listens to whom)
- `B = diag(b)` holds the exposure weights, the share of each person's update that
  comes from the message instead of from their neighbours
- `m` is the message, fixed and identical for everyone

The planner picks `b` once, subject to a budget, to minimise the discounted sum of
squared distance from the message.

## What it shows

**1. Short horizon and long horizon disagree.** On a three-person line, spending the
whole budget on the extremist beats spending it on the moderate bridge for the first
five rounds. At round 6 the ranking flips, and by round 20 the bridge is well ahead
(2.65 against 3.62). The bridge compounds; the extremist does not.

**2. The optimum is not the distance rule or the centrality rule.** On an 11-node
network with two camps:

| policy | total distance from the message |
|---|---|
| optimal | 8.48 |
| distance rule (exposure ∝ how far someone is from the message) | 10.34 |
| centrality rule (exposure ∝ eigenvector centrality) | 11.89 |
| spread evenly | 14.51 |

**3. Three things the optimum does that the naive rules do not.**

- spends **nothing** on the camp that already agrees, where the distance rule burns
  11% of the budget
- gives the bridge **3x** the exposure that either naive rule would, despite the
  bridge having the fewest connections of any node
- concentrates on two nodes instead of spreading thinly

## Layout

```
src/targeting.py   the model: dynamics, loss, optimiser, benchmark rules
src/networks.py    test networks (run it directly for the results table)
src/figures.py     regenerates everything in figures/
notebook/          Quarto lesson, renders to an HTML page or a reveal.js deck
figures/           output
```

## Running it

```bash
pip install -r requirements.txt
python src/networks.py      # the results table
python src/figures.py       # regenerate the figures

quarto render notebook/coarse_targeting.qmd --to html
quarto render notebook/coarse_targeting.qmd --to revealjs
```

## Source

Wei Li and Philip Solimine, "Targeting persistent disagreement: coarse interventions
in social networks," *Proceedings of the 27th ACM Conference on Economics and
Computation* (EC 2026).
