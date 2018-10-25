# Experiments around optimizer
### Code guidelines
* we enforce pep8
* we aim for a pylint note > 9
* 100 characters column width (not 120 nor 80)

### TODO
- [ ] change the way the enclosed face are connected (used orthogonal projection on closest edge)
- [ ] check failing floor plan : Noisy_A145, Antony_B14 (hard fails), Bussy_Regis, Massy_C102 (sotf fails)
- [ ] add way more unit tests
- [ ] add matplotlib live visualization with pytest debugging
- [ ] add load bearing walls import
- [ ] create a parallel cut (slice), useful for non rectilinear apartment
- [ ] create grid generator
- [ ] create space mutations : one face, whole border, whole perimeter etc.