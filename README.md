# Experiments around optimizer
### Objectives
* Represent an apartment blueprint by a polygon mesh using the half-edge data structure
* Generate smart mesh grids. A meaningful grid of the apartment should be able to represent 
as much viable partition of the apartment as possible. 
* Generate walls and space partition of the apartment, according to grid, user specifications, 
and architectural quality via a genetic algorithm.

### Code guidelines
* we use the typing library to enable type checking and pycharm magic
* we enforce pep8
* we aim for a pylint note > 9
* we use 100 characters column width (not 120 nor 80)

### Requirements
We aim to use as few external libraries as possible. 
The project currently requires only Numpy, Shapely and Matplotlib.
We use Pytest for testing and Pylint for linting.

### TODO
- [x] ~~correct snapping and cutting methods to prevent the creation of half-edge with the same 
starting and ending vertex~~
- [x] ~~change the way the enclosed face are connected (used orthogonal projection on closest edge)~~
- [x] ~~check failing floor plan : Noisy_A145, Antony_B14 (hard fails), Bussy_Regis, Massy_C102 
(soft fails)~~
- [x] ~~add load bearing walls import (note some plan fail due to improper geometry : duct overlapping 
load bearing wall)~~
- [x] ~~create grid generator~~
- [ ] add way more unit tests (work in progress)
- [ ] add matplotlib live visualization with pytest debugging
- [ ] create a parallel cut (slice), useful for non rectilinear apartment
- [ ] add space generation from seed growth (the idea is to generate space by growing them from a seed point)
- [ ] create space mutations : one face, whole border, whole perimeter etc.~
- [ ] create space cost functions
- [ ] create a crossover function enabling to mix two blueprints for the same apartment
- [ ] create a repair function : that will insure correct apartment circulation
- [ ] create genetic algorithm framework : population generation, random crossovers, random mutations, 
repair population, cost function evaluation, selection of new generation