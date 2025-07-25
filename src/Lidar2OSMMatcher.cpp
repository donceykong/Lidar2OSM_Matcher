
// VoxelSize used is 0.3



kiss_matcher::KeypointPair KISSMatcher::match(const std::vector<Eigen::Vector3f> &src,
                                              const std::vector<Eigen::Vector3f> &tgt) {
  clear();
  auto processInput = [&](const std::vector<Eigen::Vector3f> &input_cloud) {
    if (config_.use_voxel_sampling_) {
      return VoxelgridSampling(input_cloud, config_.voxel_size_);
    }
    return input_cloud;
  };

  auto t_init = std::chrono::high_resolution_clock::now();

  src_processed_ = std::move(processInput(src));
  tgt_processed_ = std::move(processInput(tgt));

  auto t_process = std::chrono::high_resolution_clock::now();

  faster_pfh_->setInputCloud(src_processed_);
  // Note(hlim) Some erroneous points are filtered out
  // Thus, # of `src_keypoints_` <= `src_processed_`
  faster_pfh_->ComputeFeature(src_keypoints_, src_descriptors_);

  faster_pfh_->setInputCloud(tgt_processed_);
  // Note(hlim) Some erroneous points are filtered out
  // Thus, # of `tgt_keypoints_` <= `tgt_processed_`
  faster_pfh_->ComputeFeature(tgt_keypoints_, tgt_descriptors_);

  auto t_mid = std::chrono::high_resolution_clock::now();

  const auto &corr = robin_matching_->establishCorrespondences(src_keypoints_,
                                                               tgt_keypoints_,
                                                               src_descriptors_,
                                                               tgt_descriptors_,
                                                               config_.robin_mode_,
                                                               config_.tuple_scale_,
                                                               config_.use_ratio_test_);

  src_matched_.resize(corr.size());
  tgt_matched_.resize(corr.size());

  for (size_t i = 0; i < corr.size(); ++i) {
    auto src_idx    = std::get<0>(corr[i]);
    auto dst_idx    = std::get<1>(corr[i]);
    src_matched_[i] = src_keypoints_[src_idx];
    tgt_matched_[i] = tgt_keypoints_[dst_idx];
  }

  src_matched_public_ = src_matched_;
  tgt_matched_public_ = tgt_matched_;

  auto t_end = std::chrono::high_resolution_clock::now();