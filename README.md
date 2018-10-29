# Experiments around optimizer
### Code guidelines
* we use the typing library to enable type checking an pycharm magic
* we enforce pep8
* we aim for a pylint note > 9
* we use 100 characters column width (not 120 nor 80)

### TODO
- [ ] change the way the enclosed face are connected (used orthogonal projection on closest edge)
- [x] check failing floor plan : Noisy_A145, Antony_B14 (hard fails), Bussy_Regis, Massy_C102 
(soft fails)
- [ ] add way more unit tests (work in progress)
- [ ] add matplotlib live visualization with pytest debugging
- [x] add load bearing walls import (note some plan fail due to improper geometry : duct overlapping 
load bearing wall)
- [ ] create a parallel cut (slice), useful for non rectilinear apartment
- [ ] create grid generator
- [ ] create space mutations : one face, whole border, whole perimeter etc.