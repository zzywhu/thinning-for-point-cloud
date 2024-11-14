#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/icp.h> // icp算法
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <boost/thread/thread.hpp>
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include <yaml-cpp/yaml.h>
#include "opencv2/highgui.hpp"
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/search/flann_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/bilateral.h> 
#include <pcl/common/distances.h>	
#include<omp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/geometry.h>
#include <pcl/surface/mls.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/voxel_grid.h>
#include <unordered_set>
#include <tuple>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include <unordered_map>
#include <vector>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include <unordered_map>
#include <vector>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include <unordered_map>
#include <vector>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <omp.h>
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include"thinning.h"
#include<iostream>

using namespace std;
using namespace pcl;

void calregionnormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdTree, int pos, double radius, Eigen::Matrix3f& eigenVectorsPCA, double& max_distance)
{
	std::vector<int> pointIdxNKNSearch;
	std::vector<float>pointNKNSquaredDistance;
	kdTree.nearestKSearch(cloud->points[pos], radius, pointIdxNKNSearch, pointNKNSquaredDistance);
	//kdTree.nearestKSearch(cloud->points[i], radius/2, pointIdxNKNSearch, pointNKNSquaredDistance);
	//kdTree.nearestKSearch(cloud->points[i], radius/4, pointIdxNKNSearch, pointNKNSquaredDistance);
		//如果上面的值大于0，则搜索成功
	pcl::PointCloud<pcl::PointXYZ>::Ptr neighborPoints(new pcl::PointCloud<pcl::PointXYZ>);
	//提取现在点的邻近点至neighborPoints中
	pcl::copyPointCloud(*cloud, pointIdxNKNSearch, *neighborPoints);

	Eigen::Vector4f pcaCentroid;
	Eigen::Matrix3f covariance;

	Eigen::Vector3f eigenValuesPCA;

	compute3DCentroid(*neighborPoints, pcaCentroid);//计算重心
	computeCovarianceMatrixNormalized(*neighborPoints, pcaCentroid, covariance);//compute the covariance matrix of point cloud
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);//define an eigen solver
	eigenVectorsPCA = eigen_solver.eigenvectors();//compute eigen vectors
	eigenValuesPCA = eigen_solver.eigenvalues();//compute eigen values;
	max_distance = eigenValuesPCA(0) / (eigenValuesPCA(0) + eigenValuesPCA(1) + eigenValuesPCA(2));
}

void calnormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud <pcl::Normal>::Ptr& normals)
{
	pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
	kdTree.setInputCloud(cloud);
	normals->resize(cloud->size());
#pragma omp parallel for num_threads(24)
	for (int i = 0; i < cloud->points.size(); i++)
	{
		double max_distance = 0;
		Eigen::Matrix3f eigenVectorsPCA;
		std::vector<int> pointIdxNKNSearch;
		std::vector<float>pointNKNSquaredDistance;
		kdTree.nearestKSearch(cloud->points[i], 50, pointIdxNKNSearch, pointNKNSquaredDistance);
		//kdTree.nearestKSearch(cloud->points[i], radius/2, pointIdxNKNSearch, pointNKNSquaredDistance);
		//kdTree.nearestKSearch(cloud->points[i], radius/4, pointIdxNKNSearch, pointNKNSquaredDistance);
			//如果上面的值大于0，则搜索成功
		pcl::PointCloud<pcl::PointXYZ>::Ptr neighborPoints(new pcl::PointCloud<pcl::PointXYZ>);
		//提取现在点的邻近点至neighborPoints中
		pcl::copyPointCloud(*cloud, pointIdxNKNSearch, *neighborPoints);

		Eigen::Vector4f pcaCentroid;
		Eigen::Matrix3f covariance;

		Eigen::Vector3f eigenValuesPCA;

		compute3DCentroid(*neighborPoints, pcaCentroid);//计算重心
		computeCovarianceMatrixNormalized(*neighborPoints, pcaCentroid, covariance);//compute the covariance matrix of point cloud
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);//define an eigen solver
		eigenVectorsPCA = eigen_solver.eigenvectors();//compute eigen vectors
		eigenValuesPCA = eigen_solver.eigenvalues();//compute eigen values;
		max_distance = eigenValuesPCA(0) / (eigenValuesPCA(0) + eigenValuesPCA(1) + eigenValuesPCA(2));

		pcl::Normal temp;
		temp.curvature = max_distance;
		temp.normal[0] = eigenVectorsPCA(0); temp.normal[1] = eigenVectorsPCA(1); temp.normal[2] = eigenVectorsPCA(2);
		normals->points[i] = temp;
		//temp.normal_x = eigenVectorsPCA(0); temp.normal_y = eigenVectorsPCA(1); temp.normal_z = eigenVectorsPCA(2);
	}
}

void plane_seg(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outputcloud, std::vector<pcl::PointIndices>& clusters, std::vector<Eigen::Vector3f>& plane_normals, int neighbor_threshold, double SmoothnessThreshold, double CurvatureThreshold, double distance1, double distance2)
{
	YAML::Node config = YAML::LoadFile("param/parameters.yaml");
	int regneigh = config["regneigh"].as<int>();
	// Normal estimation step
	pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	calnormal(cloud, normals);

	// Region growing algorithm parameters
	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
	reg.setMinClusterSize(800);
	reg.setMaxClusterSize(100000000);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(regneigh);
	reg.setInputCloud(cloud);
	reg.setInputNormals(normals);
	reg.setSmoothnessThreshold(SmoothnessThreshold / 180.0 * M_PI);
	reg.setCurvatureThreshold(CurvatureThreshold);

	// Extract clusters
	reg.extract(clusters);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
	*outputcloud = *colored_cloud;

	//pcl::io::savePCDFile<pcl::PointXYZRGB>("seg1.pcd", *colored_cloud);
	// Extract plane normals
	plane_normals.resize(clusters.size());
	std::vector<Eigen::Vector3f> plane_centers(clusters.size());

#pragma omp parallel for schedule(static)
	for (int i = 0; i < clusters.size(); ++i) {
		const auto& cluster_indices = clusters[i].indices;

		std::vector<Eigen::Vector3f> points;
		for (const auto& idx : cluster_indices) {
			points.push_back(Eigen::Vector3f(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z));
		}

		Eigen::MatrixXf data(points.size(), 3);
		for (size_t j = 0; j < points.size(); ++j) {
			data.row(j) = points[j];
		}

		Eigen::Vector3f mean = data.colwise().mean();
		Eigen::MatrixXf centered = data.rowwise() - mean.transpose();
		Eigen::MatrixXf covariance = (centered.transpose() * centered) / double(data.rows() - 1);
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(covariance);
		plane_normals[i] = solver.eigenvectors().col(0);
		plane_centers[i] = mean;
	}

	// Extract outliers
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
	pcl::PointIndices::Ptr outliers(new pcl::PointIndices());

	for (const auto& cluster : clusters) {
		inliers->indices.insert(inliers->indices.end(), cluster.indices.begin(), cluster.indices.end());
	}

	extract.setInputCloud(cloud);
	extract.setIndices(inliers);
	extract.setNegative(true);
	extract.filter(outliers->indices);

	size_t total_outliers = outliers->indices.size();
	size_t progress_interval = total_outliers / 10; // Update progress every 10%

	std::cout << "Processing outliers: [";
	std::cout.flush();

	std::vector<std::unordered_map<int, std::vector<int>>> local_assignments(omp_get_max_threads());
	std::vector<std::unordered_map<int, pcl::PointXYZRGB>> local_colors(omp_get_max_threads());

	// Atomic variable for tracking progress
	std::atomic<size_t> processed_outliers(0);

#pragma omp parallel
	{
		std::vector<std::pair<int, float>> closest_planes; // Stores closest plane index and vertical distance

#pragma omp for schedule(dynamic)
		for (int i = 0; i < total_outliers; ++i) {
			int idx = outliers->indices[i];
			int thread_id = omp_get_thread_num();
			Eigen::Vector3f point(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z);

			closest_planes.clear();
			for (size_t j = 0; j < clusters.size(); ++j) {
				Eigen::Vector3f normal = plane_normals[j];
				Eigen::Vector3f plane_center = plane_centers[j];
				float vertical_distance = std::abs((normal.dot(point - plane_center))) / normal.norm();
				closest_planes.push_back(std::make_pair(j, vertical_distance));
			}

			std::sort(closest_planes.begin(), closest_planes.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
				return a.second < b.second;
				});

			closest_planes.resize(5); // Keep only the closest 10 planes
			int nearest_plane_index = -1;
			float min_distance = std::numeric_limits<float>::max();
			int neighbor_count = 0;

			for (const auto& plane : closest_planes) {
				int j = plane.first;
				for (int plane_idx = 0; plane_idx < clusters[j].indices.size(); plane_idx += 10) {
					float distance = pcl::euclideanDistance(cloud->points[idx], cloud->points[clusters[j].indices[plane_idx]]);
					distance = 0.3 * distance + 0.7 * plane.second;
					if (distance < min_distance) {
						min_distance = distance;
						nearest_plane_index = j;
					}
				}
			}

			neighbor_count = clusters[nearest_plane_index].indices.size();

			if (nearest_plane_index != -1 && neighbor_count >= neighbor_threshold && min_distance < distance1) {
				local_assignments[thread_id][nearest_plane_index].push_back(idx);
				local_colors[thread_id][idx] = pcl::PointXYZRGB(
					outputcloud->points[clusters[nearest_plane_index].indices[0]].r,
					outputcloud->points[clusters[nearest_plane_index].indices[0]].g,
					outputcloud->points[clusters[nearest_plane_index].indices[0]].b
				);
			}

			// Update progress periodically
			size_t processed = processed_outliers.fetch_add(1) + 1;
			if (processed % progress_interval == 0) {
#pragma omp critical
				{
					float percent_complete = (static_cast<float>(processed) / total_outliers) * 100.0f;
					std::cout << "\rProcessing outliers: " << std::fixed << std::setprecision(2) << percent_complete << "%";
					std::cout.flush();
				}
			}
		}
	}

	std::cout << " Done!" << std::endl;

	// Update clusters and colors from local assignments and colors
	for (size_t t = 0; t < local_assignments.size(); ++t) {
		for (const auto& pair : local_assignments[t]) {
			int plane_index = pair.first;
			clusters[plane_index].indices.insert(clusters[plane_index].indices.end(), pair.second.begin(), pair.second.end());
		}

		for (const auto& color_pair : local_colors[t]) {
			outputcloud->points[color_pair.first].r = color_pair.second.r;
			outputcloud->points[color_pair.first].g = color_pair.second.g;
			outputcloud->points[color_pair.first].b = color_pair.second.b;
		}
	}
	//// Remove small clusters and set their points to red
	//std::vector<pcl::PointIndices> new_clusters;
	//for (size_t i = 0; i < clusters.size(); ++i) {
	//	if (clusters[i].indices.size() >= 2000) {
	//		// Keep clusters with enough points
	//		new_clusters.push_back(clusters[i]);
	//	}
	//	else {
	//		// Set the points in small clusters to red
	//		for (const auto& idx : clusters[i].indices) {
	//			outputcloud->points[idx].r = 255;
	//			outputcloud->points[idx].g = 0;
	//			outputcloud->points[idx].b = 0;
	//		}
	//	}
	//}

	// //Replace old clusters with filtered clusters
	//clusters = new_clusters;

	//std::cout << "Processing complete. Small clusters removed, and points colored red." << std::endl;
}


struct planeseg
{
	int R; int G; int B;
	Eigen::Vector3d eigen_normals;
	vector<int>plane;
};
atomic<int> progress(0);  // 进度条显示的进度

// 进度条显示线程函数
void display_progress(int total_work) {
	while (progress < total_work) {
		int percent = (progress * 100) / total_work;
		cout << "\rProgress: " << percent << "%" << flush;
		//this_thread::sleep_for(chrono::milliseconds(100));  // 调整显示频率
	}
	cout << "\rProgress: 100%" << endl;
}

void thinning_seg(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud2, pcl::PointCloud<pcl::PointXYZ>::Ptr& outputcloud2)
{
	YAML::Node config = YAML::LoadFile("param/parameters.yaml");
	int min_size = config["minplane_size"].as<int>();
	int thi_size = config["thinning_size"].as<int>();
	double smooth= config["SmoothnessThreshold"].as<double>();
	double cur = config["CurvatureThreshold"].as<double>();
	double smooth_2 = config["SmoothnessThreshold_2"].as<double>();
	double cur_2 = config["CurvatureThreshold_2"].as<double>();

	// 初始化变量
	std::vector<Eigen::Vector3f> plane_normals;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud < pcl::PointXYZRGB>);
	std::vector <pcl::PointIndices> clusters;
	cout << "开始平面分割..." << endl;
	// 平面分割，获取平面法向量
	plane_seg(cloud2, cloud, clusters, plane_normals, 300, smooth, cur, 0.15, 0.8);
	cout << "平面分割结束，开始薄化点云" << endl;

	// 减少不必要的文件IO操作，按需保存
	//pcl::io::savePCDFile<pcl::PointXYZRGB>("seg2.pcd", *cloud);

	pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud(new pcl::PointCloud<pcl::PointXYZ>);
	outputcloud->height = cloud->height;
	outputcloud->width = cloud->width;
	outputcloud->points.resize(cloud->size());

	for (int i = 0; i < cloud->points.size(); i++) {
		outputcloud->points[i].x = cloud->points[i].x;
		outputcloud->points[i].y = cloud->points[i].y;
		outputcloud->points[i].z = cloud->points[i].z;
	}

	vector<planeseg> plane;
	plane.reserve(clusters.size()); // 预分配内存

	for (const auto& cluster : clusters) {
		planeseg temp;
		temp.R = cloud->points[cluster.indices[0]].r;
		temp.G = cloud->points[cluster.indices[0]].g;
		temp.B = cloud->points[cluster.indices[0]].b;
		temp.plane.reserve(cluster.indices.size());
		temp.plane.insert(temp.plane.end(), cluster.indices.begin(), cluster.indices.end());
		plane.push_back(temp);
	}

	// 启动进度条显示线程
	boost::thread progress_thread(display_progress, plane.size());

	// 使用 OpenMP 并行化处理
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < plane.size(); i++) {
			// 提取平面的邻近点
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr neighborPoints(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::copyPointCloud(*cloud, plane[i].plane, *neighborPoints);
			Eigen::Vector3d eigen_normals(plane_normals[i][0], plane_normals[i][1], plane_normals[i][2]);

			pcl::KdTreeFLANN<pcl::PointXYZRGB> kdTree;
			kdTree.setInputCloud(neighborPoints);

#pragma omp parallel for
			for (int c = 0; c < plane[i].plane.size(); c++) {
				std::vector<int> pointIdxNKNSearch;
				std::vector<float> pointNKNSquaredDistance;
				Eigen::Vector3d curPoint1;
				int k = plane[i].plane[c];
				curPoint1.x() = cloud->points[k].x;
				curPoint1.y() = cloud->points[k].y;
				curPoint1.z() = cloud->points[k].z;

				// 执行 K 邻近搜索
				kdTree.nearestKSearch(cloud->points[k], thi_size, pointIdxNKNSearch, pointNKNSquaredDistance);

				Eigen::Vector3d meanPoint = Eigen::Vector3d::Zero();

				// 计算邻近点的质心
				for (const auto& idx : pointIdxNKNSearch) {
					meanPoint.x() += neighborPoints->points[idx].x;
					meanPoint.y() += neighborPoints->points[idx].y;
					meanPoint.z() += neighborPoints->points[idx].z;
				}
				meanPoint /= pointIdxNKNSearch.size();

				// 更新当前点的坐标
				curPoint1 += eigen_normals * (meanPoint - curPoint1).dot(eigen_normals);

				// 更新输出点云
				outputcloud->points[k].x = curPoint1.x();
				outputcloud->points[k].y = curPoint1.y();
				outputcloud->points[k].z = curPoint1.z();
			}

			// 更新进度条
			progress++;
		}
	}

	// 等待进度条显示线程结束
	progress_thread.join();

	// 准备输出点云
	outputcloud2 = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
	outputcloud2->height = outputcloud->height;
	outputcloud2->width = outputcloud->width;
	outputcloud2->points.resize(outputcloud->size());

#pragma omp parallel for
	for (int i = 0; i < outputcloud->points.size(); i++) {
		outputcloud2->points[i].x = outputcloud->points[i].x;
		outputcloud2->points[i].y = outputcloud->points[i].y;
		outputcloud2->points[i].z = outputcloud->points[i].z;
	}

	cout << "薄化结束，开始第二次平面分割" << endl;
	// 法线估计和第二次平面分割
	pcl::search::Search<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator2;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud3(new pcl::PointCloud < pcl::PointXYZRGB>);
	std::vector <pcl::PointIndices> clusters2;
	std::vector<Eigen::Vector3f> plane_normals2;
	//pcl::io::savePCDFile<pcl::PointXYZ>("check.pcd", *outputcloud2);
	plane_seg(outputcloud2, cloud3, clusters2, plane_normals2, 1000, smooth_2, cur_2, 0.25, 1.0);
	cout << "平面分割结束，开始精细化点云" << endl;
	// 保存和输出优化后的点云
	//pcl::io::savePCDFile<pcl::PointXYZRGB>("check.pcd", *cloud3);

	outputcloud2->height = cloud3->height;
	outputcloud2->width = cloud3->width;
	outputcloud2->points.resize(cloud3->size());

#pragma omp parallel for
	for (int i = 0; i < cloud3->points.size(); i++) {
		outputcloud2->points[i].x = cloud3->points[i].x;
		outputcloud2->points[i].y = cloud3->points[i].y;
		outputcloud2->points[i].z = cloud3->points[i].z;
	}

	// 调用精细化平面分割
	//plane_seg_update(outputcloud2, outputcloud2, clusters2, plane_normals2, 1000);
	plane_seg_refine(outputcloud2, outputcloud2, clusters2, plane_normals2, min_size);
	cout << "finish thinning！" << endl;
}



void randomSamplePointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& inputCloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& outputCloud,
	float sampleRate)
{
	pcl::RandomSample<pcl::PointXYZ> randomSampler;
	randomSampler.setInputCloud(inputCloud);
	randomSampler.setSample(static_cast<unsigned int>(inputCloud->size() * sampleRate));
	randomSampler.filter(*outputCloud);
}


int main()
{
	YAML::Node config = YAML::LoadFile("param/parameters.yaml");
	string filename = config["data_path"].as<std::string>();
	int isdown = config["ifdown"].as<int>();
	double downsize = config["downsize"].as<double>();
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_seg(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr outcloud_seg(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<Eigen::Vector3f> plane_normals;
	std::vector <pcl::PointIndices>clusters;
	std::vector<std::vector<int>> clusters_index;
	pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud);
	if (isdown == 1)
	{
		cout << "start downsample" << endl;
		randomSamplePointCloud(cloud, cloud, downsize);
	}
	cleanPointCloud(cloud, cloud);
	pcl::io::savePCDFile<pcl::PointXYZ>("result/down.pcd", *cloud);
	thinning_seg(cloud, cloud);
	pcl::io::savePCDFile<pcl::PointXYZ>("result/result.pcd", *cloud);
	return 1;
}

