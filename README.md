# dnd-spell-generator
Using Recurrent Neural Nets to generate DnD spells

## TODO

Scrape [this](https://www.dnd-spells.com/spells) site and use its text with a character-by-character RNN (Perhaps the one [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)) to generate new (and hopefully valid, though probably not balanced) spells. Once that is working see if you can somehow add a balance term to the loss to make balanced spells.

Heck maybe even use this to create new classes or subclasses or stat blocks, let your imagination go wild!

## References
I used:
* https://github.com/tatp22/multidim-positional-encoding which has an MIT license
* https://github.com/wangleiofficial/label-smoothing-pytorch which has an MIT license
    * I had to rename the module and make a few changes to get it to work though.
* https://github.com/minimaxir/gpt-2-simple which has an MIT license
* I also borrowed heavily from the previous labs, but tried to understand all the parts I used, hence why it took so long to code up.