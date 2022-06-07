output_dir="saved_model"
# use ranklib to evaluate prediction results
python ./sort_by_score.py -res_dir $output_dir -epoch model_1.pt
for metric in p@1 map err@5
do
    java -jar ./RankLib-2.16.jar -test "${output_dir}/sorted_test_model_1.pt.dat" \
        -metric2T $metric -idv "${output_dir}/${metric}_${epoch}.txt"
done
