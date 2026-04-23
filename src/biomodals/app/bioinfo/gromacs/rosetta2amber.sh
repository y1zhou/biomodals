#! /bin/bash

# author: Ziwei Pang (Jeffery)@biomap
input="$1"

pdb4amber -i "${input}" -o input.pdb -y --dry

sed -i 's/ 1H / H1 /g' input.pdb
sed -i 's/ 2H / H2 /g' input.pdb
sed -i 's/ 3H / H3 /g' input.pdb

sed -i 's/ 1HA / HA2 /g' input.pdb
sed -i 's/ 2HA / HA3 /g' input.pdb

sed -i 's/ 1HB / HB2 /g' input.pdb
sed -i 's/ 2HB / HB3 /g' input.pdb
sed -i 's/ 3HB / HB1 /g' input.pdb

sed -i 's/ 1HG / HG2 /g' input.pdb
sed -i 's/ 2HG / HG3 /g' input.pdb

sed -i 's/ 1HD / HD2 /g' input.pdb
sed -i 's/ 2HD / HD3 /g' input.pdb

sed -i 's/ 1HE / HE2 /g' input.pdb
sed -i 's/ 2HE / HE3 /g' input.pdb
sed -i 's/ 3HE / HE1 /g' input.pdb

sed -i 's/ 1HZ / HZ1 /g' input.pdb
sed -i 's/ 2HZ / HZ2 /g' input.pdb
sed -i 's/ 3HZ / HZ3 /g' input.pdb

sed -i 's/ 1HD1 / HD11 /g' input.pdb
sed -i 's/ 2HD1 / HD12 /g' input.pdb
sed -i 's/ 3HD1 / HD13 /g' input.pdb

sed -i 's/ 1HD2 / HD21 /g' input.pdb
sed -i 's/ 2HD2 / HD22 /g' input.pdb
sed -i 's/ 3HD2 / HD23 /g' input.pdb

sed -i 's/ 1HG1 / HG12 /g' input.pdb
sed -i 's/ 2HG1 / HG13 /g' input.pdb

sed -i 's/ 1HG2 / HG21 /g' input.pdb
sed -i 's/ 2HG2 / HG22 /g' input.pdb
sed -i 's/ 3HG2 / HG23 /g' input.pdb
