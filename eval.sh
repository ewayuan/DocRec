output_dir="saved_model"
# use ranklib to evaluate prediction results
python ./sort_by_score.py -res_dir $output_dir -epoch 5_model.pt
for metric in p@1 map err@5
do
    java -jar ./RankLib-2.16.jar -test "${output_dir}/sorted_test_${epoch}_model.pt.dat" \
        -metric2T $metric -idv "${output_dir}/${metric}_${epoch}.txt"
done
