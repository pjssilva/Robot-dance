# Robot Dance

This repository contains the code used in the manuscript 
[Robot dance: a city-wise automatic control of Covid-19 mitigation levels](https://www.medrxiv.org/content/10.1101/2020.05.11.20098541v1).

The idea is to sue control and optimization to design efficient alternate mitigation
strategies for Covid-19 and other infectious diseases. It uses an adapted SEIR model that
takes into account the mobility between different cities. The objective is to design a
protocols that controls the number of infected to a maximum level while alternating between
more stringent and more light mitigation and between cities (or regions), so that the State
or country does not close all at once.

The code up to now should work for a few cities (or regions), a couple of tens. We have
some ideas to make it scale to the hundreds.

## Copyright 

Copyright Paulo J. S. Silva e Luis Gustavo Nonato. See the [license file](LICENSE.MD).

## Please Cite Us

We provide this code hoping that it will be useful for others. Please if you use it, let us
know about you experiences. Moreover, if you use this code in any publication, please cite
us. This is very important. For the moment we only have the manuscript, so cite it as

Silva, Paulo J. S., Pereira, Tiago and Nonato, Luis Gustavo. "Robot dance: a city-wise
automatic control of Covid-19 mitigation levels". Medrxiv, 2020.
[doi:https://doi.org/10.1101/2020.05.11.20098541]

