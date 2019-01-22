#***********************************************************
#Copyright 2018 eBay Inc.
#Use of this source code is governed by a MIT-style
#license that can be found in the LICENSE file or at
#https://opensource.org/licenses/MIT.
#***********************************************************
#!/usr/bin/env bash
alpha_array=("0.0")
for alpha in ${alpha_array[@]}
do
  echo "evaluate on "alpha
  python eval_score.py $alpha $1 $2
done
