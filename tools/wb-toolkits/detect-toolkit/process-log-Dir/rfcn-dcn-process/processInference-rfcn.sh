#!/bin/bash
# set -x
if [ ! -n "$1" ]
then
    echo "must input file path"
    exit
else
    echo "the input file is : "$1
fi
scriptFile="/workspace/data/BK/rfcn/Deformable-ConvNets/rfcn/rfcn_dcn_inference.py"
inputFile=$1
splitFilePrefix="split_file-"
# gpu id  used to run
gpuArray=(0 1 2 3 4 5 6 7)
#gpuArray=(5 6 7)
gpuNum=${#gpuArray[@]}
# inputfile
inputBasePath=`dirname $inputFile`
inputFileName=`basename $inputFile`
# split the input file by gpuNum
inputFileLineCount=`wc -l $inputFile|awk '{print $1}'`
perFileLineCount=`expr $inputFileLineCount / $gpuNum`
perFileLineCount_flag=`expr $inputFileLineCount % $gpuNum`
if [ $perFileLineCount_flag -ne 0 ]
then
    perFileLineCount=`expr $perFileLineCount + 1`
fi
echo "perFileLineCount : "$perFileLineCount
# split
split_output_prefix=$inputBasePath"/"$splitFilePrefix
split -l $perFileLineCount $inputFile -d -a 2 $split_output_prefix
# get split result file
splitFileArray=()
tempFileList=`ls $inputBasePath`
for file in $tempFileList
do
    if [[ $file == $splitFilePrefix* ]]
    then
        if [[ $file == *"result.json" ]] || [[ $file == *"run.log" ]] || [[ $file == *"stderr.log" ]]
        then
            echo "don't process the file : "$file
        else
            tempFile=$inputBasePath"/"$file
            splitFileArray=(${splitFileArray[*]} $tempFile)
        fi
    fi
done

# check
if [ $gpuNum -ne ${#splitFileArray[@]} ]
then
    echo "ERROR : log file count unequal gpu count"
    exit
fi

for(( i=0;i<$gpuNum;i++))
do
    echo "gpu id is : "${gpuArray[i]}
    echo "log file is : "${splitFileArray[i]}
    runCmd="nohup python -u "$scriptFile" \
    --imageListFile "${splitFileArray[i]}" \
    --urlFlag 1 \
    --gpuId "${gpuArray[i]}" \
    --outputFileFlag 2 \
    --visualizeFlag 0 \
    > "${splitFileArray[i]}"_run.log \
    2> "${splitFileArray[i]}"_stderr.log &"
    echo $runCmd
    `eval $runCmd`
done
