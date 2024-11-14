#ifndef PLANE_SEG_REFINE_H
#define PLANE_SEG_REFINE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <omp.h>
#include <random>
// Helper function to generate a random color
void generateRandomColor(uint8_t& r, uint8_t& g, uint8_t& b) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 255);

    r = static_cast<uint8_t>(dis(gen));
    g = static_cast<uint8_t>(dis(gen));
    b = static_cast<uint8_t>(dis(gen));
}

// Function to color the point cloud clusters
pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    const std::vector<pcl::PointIndices>& clusters)
{
    // Create a new PointCloud to hold the colored points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    // Iterate through each cluster
    for (const auto& cluster : clusters) {
        // Generate a random color for each cluster
        uint8_t r, g, b;
        generateRandomColor(r, g, b);

        // Assign each point in the cluster a color
        for (const auto& index : cluster.indices) {
            pcl::PointXYZRGB point;
            point.x = cloud->points[index].x;
            point.y = cloud->points[index].y;
            point.z = cloud->points[index].z;
            point.r = r;
            point.g = g;
            point.b = b;

            colored_cloud->points.push_back(point);
        }
    }

    // Set the width and height of the colored cloud
    colored_cloud->width = static_cast<uint32_t>(colored_cloud->points.size());
    colored_cloud->height = 1; // Unorganized point cloud
    colored_cloud->is_dense = true;

    return colored_cloud;
}

void cleanPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud2) {
    // 确保输出点云是空的
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud(new pcl::PointCloud<pcl::PointXYZ>);
    outputcloud->clear();
    outputcloud->reserve(cloud->points.size()); // 预留空间，避免频繁分配

    // 使用 OpenMP 并行化遍历点云中的每个点
#pragma omp parallel
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);

#pragma omp for nowait
        for (int i = 0; i < cloud->points.size(); ++i) {
            if (!(cloud->points[i].x == 0.0f && cloud->points[i].y == 0.0f && cloud->points[i].z == 0.0f)) {
                local_cloud->points.push_back(cloud->points[i]);
            }
        }

#pragma omp critical
        {
            outputcloud->points.insert(outputcloud->points.end(), local_cloud->points.begin(), local_cloud->points.end());
        }
    }

    // 更新输出点云的大小信息
    outputcloud->width = static_cast<uint32_t>(outputcloud->points.size());
    outputcloud->height = 1; // 因为点云是无序的，所以设置高度为1
    outputcloud->is_dense = true;
    outputcloud2->height = outputcloud->height;
    outputcloud2->width = outputcloud->width;
    outputcloud2->points.resize(outputcloud->size());

#pragma omp parallel for
    for (int i = 0; i < outputcloud->points.size(); i++) {
        outputcloud2->points[i].x = outputcloud->points[i].x;
        outputcloud2->points[i].y = outputcloud->points[i].y;
        outputcloud2->points[i].z = outputcloud->points[i].z;
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr generateLineCloud(const Eigen::Vector3f& start_point,
    const Eigen::Vector3f& direction,
    float length,
    int num_points) {
    // 创建一个空的点云对象
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // 归一化方向向量
    Eigen::Vector3f unit_direction = direction.normalized();

    // 计算每个点的间隔
    float step_size = length / static_cast<float>(num_points - 1);

    // 生成线段上的点
    for (int i = 0; i < num_points; ++i) {
        // 计算当前点的位置：起点 + 方向 * 当前步数
        Eigen::Vector3f point = start_point + unit_direction * step_size * i;
        cloud->points.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }

    // 设置点云的宽和高
    cloud->width = cloud->points.size();
    cloud->height = 1;  // 这是一个无序的点云，因此高度为 1
    cloud->is_dense = true;

    return cloud;
}


// 分类并将不符合条件的点坐标设为0
void classify_and_zero_points(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud,
    const std::vector<int>& indices,
    const Eigen::Vector3f& normal,
    const Eigen::Vector3f& center,
    float proportion_threshold
) {
    std::vector<int> positive_class;
    std::vector<int> negative_class;

    // 计算初始分类
    for (int idx : indices) {
        const auto& point = cloud->points[idx];
        Eigen::Vector3f point_vec(point.x, point.y, point.z);
        Eigen::Vector3f vec_to_center = point_vec - center;
        float projection = vec_to_center.dot(normal);

        if (projection > 0) {
            positive_class.push_back(idx);
        }
        else {
            negative_class.push_back(idx);
        }
    }


    // 如果某一类点数占比小于另一类，调整阈值重新分类
    double adjustment = 0.0;
    if (positive_class.size() < negative_class.size()) {
        adjustment = 0.005;  // 放宽 negative 的阈值
    }
    if (negative_class.size() < positive_class.size()) {
        adjustment = -0.005;   // 放宽 positive 的阈值
    }

    // 重新分类
    positive_class.clear();
    negative_class.clear();
    for (int idx : indices) {
        const auto& point = cloud->points[idx];
        Eigen::Vector3f point_vec(point.x, point.y, point.z);
        Eigen::Vector3f vec_to_center = point_vec - center;
        float projection = vec_to_center.dot(normal);

        // 调整后的阈值
        if (projection > adjustment) {
            positive_class.push_back(idx);
        }
        else {
            negative_class.push_back(idx);
        }
    }

    // 判断哪一类的点数占比小于阈值，将这些点设为0
    if (static_cast<float>(positive_class.size()) / indices.size() < proportion_threshold) {
        for (int idx : positive_class) {
            cloud->points[idx].x = 0;
            cloud->points[idx].y = 0;
            cloud->points[idx].z = 0;
        }
    }
    else if (static_cast<float>(negative_class.size()) / indices.size() < proportion_threshold) {
        for (int idx : negative_class) {
            cloud->points[idx].x = 0;
            cloud->points[idx].y = 0;
            cloud->points[idx].z = 0;
        }
    }
    else {
        *outputcloud = *cloud;
    }
    *outputcloud = *cloud;
}

void extend_plane(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud,
    std::vector<int>& indices,  // 修改：确保 indices 是可以修改的
    Eigen::Vector3f& normal,
    Eigen::Vector3f& center,
    float move_distance, // 需要移动的距离
    float proximity_threshold // 判断距离另一平面近的阈值
) {
    std::vector<int> positive_class;
    std::vector<int> negative_class;

    // 计算初始分类
    for (int idx : indices) {
        const auto& point = cloud->points[idx];
        Eigen::Vector3f point_vec(point.x, point.y, point.z);
        Eigen::Vector3f vec_to_center = point_vec - center;
        float projection = vec_to_center.dot(normal);

        if (projection > 0) {
            positive_class.push_back(idx);
        }
        else {
            negative_class.push_back(idx);
        }
    }

    float positive_ratio = static_cast<float>(positive_class.size()) / indices.size();
    float negative_ratio = static_cast<float>(negative_class.size()) / indices.size();

    // 如果某一类点数占比小于另一类，调整阈值重新分类
    double adjustment = 0.0;
    if (positive_ratio < negative_ratio) {
        adjustment = 0.0;  // 放宽 negative 的阈值
    }
    else if (negative_ratio < positive_ratio) {
        adjustment = -0.0;   // 放宽 positive 的阈值
    }

    // 重新分类
    positive_class.clear();
    negative_class.clear();
    for (int idx : indices) {
        const auto& point = cloud->points[idx];
        Eigen::Vector3f point_vec(point.x, point.y, point.z);
        Eigen::Vector3f vec_to_center = point_vec - center;
        float projection = vec_to_center.dot(normal);

        // 调整后的阈值
        if (projection > adjustment) {
            positive_class.push_back(idx);
        }
        else {
            negative_class.push_back(idx);
        }
    }

    // 判断哪一个类别占比较大
    std::vector<int>* larger_class;
    std::vector<int>* smaller_class;
    if (positive_class.size() > negative_class.size()) {
        larger_class = &positive_class;
        smaller_class = &negative_class;
    }
    else {
        larger_class = &negative_class;
        smaller_class = &positive_class;
    }

    // 对较大的类进行点的移动操作
    Eigen::Vector3f new_center(0, 0, 0); int numcount = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr neighborPoints(new pcl::PointCloud<pcl::PointXYZ>);
    neighborPoints->height = 1; neighborPoints->width = 0;
    for (int idx : *larger_class) {
        const auto& point = cloud->points[idx];
        Eigen::Vector3f point_vec(point.x, point.y, point.z);
        Eigen::Vector3f vec_to_center = point_vec - center;
        float distance_to_plane = vec_to_center.dot(normal); // 计算该点到另一个平面的距离

        // 判断点是否离另一个平面较近
        if (std::abs(distance_to_plane) < proximity_threshold && distance_to_plane > adjustment) {
            // 沿着法向移动设定的距离
            Eigen::Vector3f new_point_vec = point_vec - normal * move_distance;
            pcl::PointXYZ new_point(new_point_vec.x(), new_point_vec.y(), new_point_vec.z());

            // 加入新点到点云中，并且将新点的索引加入到 indices 中，保留原点
#pragma omp critical
            {
                Eigen::Vector3f point_center_new(new_point_vec.x(), new_point_vec.y(), new_point_vec.z());
                new_center += point_center_new; numcount++;
                neighborPoints->push_back(new_point);
                neighborPoints->width++;
                cloud->points.push_back(new_point);
                cloud->width++;
                indices.push_back(cloud->points.size() - 1);  // 将新点的索引加入 indices
            }
        }
        // 判断点是否离另一个平面较近
        if (std::abs(distance_to_plane) < proximity_threshold && distance_to_plane < adjustment) {
            // 沿着法向移动设定的距离
            Eigen::Vector3f new_point_vec = point_vec + normal * move_distance;
            pcl::PointXYZ new_point(new_point_vec.x(), new_point_vec.y(), new_point_vec.z());

            // 加入新点到点云中，并且将新点的索引加入到 indices 中，保留原点
#pragma omp critical
            {
                Eigen::Vector3f point_center_new(new_point_vec.x(), new_point_vec.y(), new_point_vec.z());
                new_center += point_center_new; numcount++;
                neighborPoints->push_back(new_point);
                neighborPoints->width++;
                cloud->points.push_back(new_point);
                cloud->width++;
                indices.push_back(cloud->points.size() - 1);  // 将新点的索引加入 indices
            }
        }
    }

    center = new_center / numcount;


    Eigen::Vector4f pcaCentroid;
    Eigen::Matrix3f covariance;
    Eigen::Matrix3f eigenVectorsPCA;

    //compute3DCentroid(*neighborPoints, pcaCentroid);//计算重心
    //computeCovarianceMatrixNormalized(*neighborPoints, pcaCentroid, covariance);//compute the covariance matrix of point cloud
    //Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);//define an eigen solver
    //eigenVectorsPCA = eigen_solver.eigenvectors();//compute eigen vectors
    //normal.x()= eigenVectorsPCA(0); normal.y() = eigenVectorsPCA(1); normal.z() = eigenVectorsPCA(2);
    // 输出更新后的点云
    *outputcloud = *cloud;
}



// 平面对结构体
struct PlanePair {
    int plane_idx1;
    int plane_idx2;
    float min_distance;

    PlanePair(int idx1, int idx2, float distance)
        : plane_idx1(std::min(idx1, idx2)), plane_idx2(std::max(idx1, idx2)), min_distance(distance) {}

    // 重载 < 运算符，便于将平面对插入到 set 中时避免重复
    bool operator<(const PlanePair& other) const {
        return std::tie(plane_idx1, plane_idx2, min_distance) < std::tie(other.plane_idx1, other.plane_idx2, other.min_distance);
    }

    // 检查两个平面对是否等价（保证 plane_idx1 < plane_idx2）
    bool operator==(const PlanePair& other) const {
        return plane_idx1 == other.plane_idx1 && plane_idx2 == other.plane_idx2;
    }
};

// 用于 hash 函数
struct PlanePairHash {
    size_t operator()(const PlanePair& pair) const {
        return std::hash<int>()(pair.plane_idx1) ^ std::hash<int>()(pair.plane_idx2);
    }
};

void plane_seg_refine(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputcloud,
    std::vector<pcl::PointIndices>& clusters,
    const std::vector<Eigen::Vector3f>& plane_normals,
    int min_plane_points_threshold // 平面点数阈值
) {
    int max_points = 0;
    int max_index = -1;

    // 找到点数最多的平面
    for (int i = 0; i < clusters.size(); ++i) {
        if (clusters[i].indices.size() > max_points) {
            max_points = clusters[i].indices.size();
            max_index = i;
        }
    }

    const float angle_deviation = 15.0f; // 角度偏差阈值（度）
    const float orthogonal_threshold = std::sin(angle_deviation * M_PI / 180.0f); // 互相垂直的容差
    float distance_threshold = 1; // 点到平面的最大距离（米）
    distance_threshold = distance_threshold * distance_threshold;
    std::vector<Eigen::Vector3f> plane_centers(clusters.size());

    // 用于记录平面对，确保没有重复对出现
    std::unordered_set<PlanePair, PlanePairHash> plane_pairs;

    // 计算每个平面的中心点
    for (int i = 0; i < clusters.size(); ++i) {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, clusters[i].indices, centroid);
        plane_centers[i] = centroid.head<3>();
    }

    std::vector<PlanePair> potential_pairs;  // 用于在循环中收集潜在的平面对

#pragma omp parallel for num_threads(24)
    for (int i = 0; i < clusters.size(); ++i) {
        if (clusters[i].indices.size() < min_plane_points_threshold || i == max_index) {
            continue;
        }

        std::vector<std::pair<int, float>> potential_planes; // 存储平面 j 的索引及其垂直度

        // 寻找与平面 i 近似垂直的前五个平面
        for (int j = 0; j < clusters.size(); ++j) {
            if (i == j || j == max_index || clusters[j].indices.size() < min_plane_points_threshold) {
                continue;
            }

            Eigen::Vector3f normal_i = plane_normals[i].normalized();
            Eigen::Vector3f normal_j = plane_normals[j].normalized();
            float dot_product = normal_i.dot(normal_j);

            if (std::fabs(dot_product) < orthogonal_threshold) {
                potential_planes.emplace_back(j, std::fabs(dot_product));
            }
        }

        // 按垂直度排序，选择前五个最接近垂直的平面
        std::sort(potential_planes.begin(), potential_planes.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });

        if (potential_planes.size() > 5) {
            potential_planes.resize(5);
        }

        int closest_plane = -1;
        float min_distance = std::numeric_limits<float>::max();

        // 计算平面 i 与这些平面之间的最小欧氏距离
        for (const auto& [j, _] : potential_planes) {
            for (int idx_i = 0; idx_i < clusters[i].indices.size(); idx_i += 100) {
                const auto& point_i = cloud->points[clusters[i].indices[idx_i]];
                Eigen::Vector3f point_vec_i(point_i.x, point_i.y, point_i.z);

                for (int idx_j = 0; idx_j < clusters[j].indices.size(); idx_j += 100) {
                    const auto& point_j = cloud->points[clusters[j].indices[idx_j]];
                    Eigen::Vector3f point_vec_j(point_j.x, point_j.y, point_j.z);

                    float distance = (point_vec_i - point_vec_j).norm();
                    if (distance < min_distance) {
                        min_distance = distance;
                        closest_plane = j;
                    }
                }
            }
        }

        // 添加平面对到临时集合中
        if (min_distance <= distance_threshold && closest_plane != -1) {
            PlanePair new_pair(i, closest_plane, min_distance);

#pragma omp critical
            {
                potential_pairs.push_back(new_pair);  // 收集潜在的平面对
            }
        }
    }

    // 在并行循环之外检查是否有重复的平面对并插入
    for (const auto& new_pair : potential_pairs) {
        if (plane_pairs.find(new_pair) == plane_pairs.end()) {
            plane_pairs.insert(new_pair);  // 确保无重复配对后插入
        }
    }

    std::vector<pcl::PointIndices> clusters_check;
    // 遍历平面对，并进行操作
    pcl::PointCloud<pcl::PointXYZ>::Ptr linecloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& pair : plane_pairs) {
        std::vector<int> modifiable_indices_i = clusters[pair.plane_idx1].indices;
        std::vector<int> modifiable_indices_j = clusters[pair.plane_idx2].indices;
        Eigen::Vector3f centers2 = plane_centers[pair.plane_idx2];
        Eigen::Vector3f centers1 = plane_centers[pair.plane_idx1];
        Eigen::Vector3f normal2 = plane_normals[pair.plane_idx2];
        Eigen::Vector3f normal1 = plane_normals[pair.plane_idx1];
        extend_plane(cloud, outputcloud, modifiable_indices_i, normal2, centers2, 0.15, 0.25);
        extend_plane(cloud, outputcloud, modifiable_indices_j, normal1, centers1, 0.15, 0.25);
        //std::cout << centers1 << std::endl;
        //std::cout << centers2 << std::endl;
        //std::cout << normal1 << std::endl;
        //std::cout << normal2 << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr linecloud1(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr linecloud2(new pcl::PointCloud<pcl::PointXYZ>);
        //linecloud1= generateLineCloud(centers2, normal2, 2, 1000);
        //linecloud2 = generateLineCloud(centers1, normal1, 2, 1000);
        *linecloud += *linecloud1 + *linecloud2;
        classify_and_zero_points(cloud, outputcloud, modifiable_indices_i, normal2, centers1, 0.2);
        classify_and_zero_points(cloud, outputcloud, modifiable_indices_j, normal1, centers2, 0.2);
        clusters[pair.plane_idx1].indices = modifiable_indices_i;
        clusters[pair.plane_idx2].indices = modifiable_indices_j;
        clusters_check.push_back(clusters[pair.plane_idx1]);
        clusters_check.push_back(clusters[pair.plane_idx2]);
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr check = colorClusters(outputcloud, clusters_check);
    //pcl::io::savePCDFile<pcl::PointXYZRGB>("color.pcd", *check);
    //pcl::io::savePCDFile<pcl::PointXYZ>("line.pcd", *linecloud);
}

#endif // PLANE_SEG_REFINE_H

