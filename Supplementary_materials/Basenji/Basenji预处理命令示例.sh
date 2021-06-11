#同源序列划分
python ./basenji_hdf5_synteny_training_tz_pca_kmeans_cluster.py -l 131072 -p 20 -t 0.1 -v 0.1 -o ./h5_bed/with_/131/gf_NIP24_131.bed  -g ./NIP_unmap.bed  ./NIPP.fasta  ./NIP_ATAC_merge_bw_list  ./h5_bed/with_/131/gf_NIP24_131.h5 ./all_pfam_gene_and_gene_family_list.bed 

python ./basenji_hdf5_synteny_training_tz_pca_kmeans_cluster.py -l 16384 -p 20 -t 0.1 -v 0.1 -o ./h5_bed/with_/16/gf_NIP24_16.bed  -g ./NIP_unmap.bed  ./NIPP.fasta  ./NIP_ATAC_merge_bw_list  ./h5_bed/with_/16/gf_NIP24_16.h5 ./all_pfam_gene_and_gene_family_list.bed 


#无同源序列划分
python basenji_hdf5_single_tz.py -l 131072  -p 20 -t 0.1 -v 0.1 -o ./h5_bed/without_/131/NIP24_131.bed   -g  ./NIP_unmap.bed  ./NIPP.fasta ./NIP_ATAC_merge_bw_list  ./h5_bed/without_/131/NIP24_131.h5

python basenji_hdf5_single_tz.py -l 16384  -p 20 -t 0.1 -v 0.1 -o ./h5_bed/without_/16/NIP24_16.bed   -g  ./NIP_unmap.bed  ./NIPP.fasta ./NIP_ATAC_merge_bw_list  ./h5_bed/without_/16/NIP24_16.h5

#注：同源序列划分和修改版的Basenji预处理代码由涂师兄实现，这一点在本人毕业论文的致谢部分有体现。


