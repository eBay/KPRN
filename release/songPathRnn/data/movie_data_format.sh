#!/usr/bin/env bash
python movie_data_format.py -i input -d output -o 0 -g 0 -e 0 -m 6 -t 1


data_set=("train" "test")
int2torch="th int2torch.lua"
movie_out_dir="output"
for set in "${data_set[@]}"
do
	echo $set
	dataDir=${movie_out_dir}/${set}
	dataset=$set
	if [ -f ${movie_out_dir}/${dataset}.list ]; then
		rm ${movie_out_dir}/${dataset}.list #the downstream training code reads in this list of filenames of the data files, split by length
	fi
	echo "converting $dataset to torch files"
	for ff in $dataDir/*.int
	do
	    echo $ff
		out=`echo $ff | sed 's|.int$||'`.torch
		$int2torch -input $ff -output $out -tokenLabels 0 -tokenFeatures 1 -addOne 1 #convert to torch format
		if [ $? -ne 0 ]
		then
			echo 'int2torch failed!'
			echo 'Failed for relation'
			continue #continue to the next one
		fi
	done
done

data_set=("train" "test")

for set in "${data_set[@]}"
do
	echo $set
	dataDir=${movie_out_dir}/${set}
	for f in `ls $dataDir/*.torch`
	do
		cmd="th insertClassLabels.lua -input $f -classLabel 1"
		$cmd
	done
done

python movie_data_list.py movie_output_data/rate

python format_entity_pair.py "test.list"