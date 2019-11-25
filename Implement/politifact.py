from politifact import Politifact

p = Politifact()
obama_related = p.statements().people('Barack Obama')

for statement in obama_related:
  print(statement.ruling)

# OUTPUT:
# > True
# > Pants On Fire!
# > Mostly True
# > ...