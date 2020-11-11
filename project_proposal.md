# Deep Learning Final Project Proposal

## Goal
Use deep learning techniques to generate viable content for the table top role playing game Dungeons and Dragons.
Initially the goal will be to generate spells similar to those used in the game. 
The goal is to not only generate text that is similar in style and form to the ones used in the game, but also generating content that is viable from a gameplay perspective (i.e. could be used within the game rules) as well as hopefully being balanced (i.e. able to be used in a game without severely advantaging the user in some way compared to others).
If generating spells proves infeasible or is easy, other options include generating non-player characters, monsters, or descriptions of places.

## Approach

The initial approach will be to use GURs and/or LSTMs, most likely using an initial architecture similar to Google's Neural Machine Translation architecture.
If it makes more sense or the previous method is not viable, I will switch to using transformer architectures.
Finally, I will attempt to learn more about and train a custom BERT model.
I will develop custom loss functions attempting to optimize the validity and balance of generated content as well, although this will be one of the more difficult parts of the project.

## Measures of Success

Most measures of success in this context will be heuristic at best, as the goal itself is subjective.
Certain techniques such as regexes (for finding key parts of the document or locating necessary formatting) and topic modeling can provide empirical data, but this is insufficient.
The first measure of success, therefore will be my own judgement, especially when it comes to measuring game balance.
Next in line are a group of friends I have that play the game regularly and are familiar with the rules and common practices and will be able to provide feedback.
Finally, I plan to use several communities on the website reddit.com in order to give a more unbiased analysis of the generated content.
These communities consist of regular players of the game as well as amateur (and in some cases professional) content creators for the game, and should be able to give a good measurement of the success of the project.

## Available Resources

I own digital copies of the content I intend to use as training data.
Spell data is also available on [this website](https://www.dnd-spells.com/spells) and is easy to scrape.
Additionally, I could acquire content from several Dungeons and Dragons content marketplaces, such as [this](https://www.dmsguild.com/).
The aformentioned reddit comunities also provide a vetted source of possible training data, as some of the groups specialize in producing the same kind of content I am trying to generate.