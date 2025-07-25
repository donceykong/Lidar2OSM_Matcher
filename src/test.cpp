#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/io/pcd_io.h>

int main()
{
    // Load or create point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("peer_global_map.pcd", *cloud);

    // // Visualize cloud
    // pcl::visualization::PCLVisualizer viewer1("Simple Cloud Viewer");
    // viewer1.setBackgroundColor(0.0, 1.0, 1.0);
    // viewer1.addPointCloud<pcl::PointXYZ>(cloud, "pc");

    // while (!viewer1.wasStopped()) {
    //     viewer1.spin();
    // }

    // Estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud(cloud);
    normal_estimation.setRadiusSearch(2.0);  // or use setKSearch(k)
    normal_estimation.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
    normal_estimation.compute(*normals);

    // Compute FPFH
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
    fpfh.setRadiusSearch(5);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh.compute(*fpfh_features);

    // std::cout << "Computed " << fpfh_features->size() << " FPFH descriptors." << std::endl;
    // std::cout << "Each descriptor has 33 bins." << std::endl;
    // if (!fpfh_features->empty()) {
    //     std::cout << "First FPFH histogram: " << std::endl;
    //     for (int i = 0; i < fpfh_features->size()-1; i++) {
    //         std::cout << "point " << i << ": ";
    //         for (int j = 0; j < 33; ++j) {
    //             std::cout << fpfh_features->points[i].histogram[j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "FPFH feature dimension: " << fpfh_features->points[0].histogram[0] << std::endl;

    // Create RGB point cloud for visualization
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->resize(cloud->size());

    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];
        const auto& hist = fpfh_features->points[i].histogram;

        // Copy histogram to vector and normalize
        std::vector<float> h(hist, hist + 33);
        float sum = std::accumulate(h.begin(), h.end(), 0.0f);
        if (sum > 1e-5f) {
            for (auto& v : h) v /= sum;
        }

        // Sort indices by descending value
        std::vector<int> indices(33);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return h[a] > h[b];
        });

        // Map top 3 bins to RGB
        float r = h[indices[0]] * 255.0f;
        float g = h[indices[1]] * 255.0f;
        float b = h[indices[2]] * 255.0f;

        pcl::PointXYZRGB rgb_pt;
        rgb_pt.x = pt.x;
        rgb_pt.y = pt.y;
        rgb_pt.z = pt.z;
        rgb_pt.r = static_cast<uint8_t>(std::min(r, 255.0f));
        rgb_pt.g = static_cast<uint8_t>(std::min(g, 255.0f));
        rgb_pt.b = static_cast<uint8_t>(std::min(b, 255.0f));
        colored_cloud->points[i] = rgb_pt;
    }

    // Visualize colored point cloud
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("FPFH Color Viewer"));
    viewer->setBackgroundColor(0.8, 0.8, 0.8);
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, "fpfh_rgb");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "fpfh_rgb");
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spin();
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
