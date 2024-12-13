#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>  
#include <memory> 
#include <unordered_set>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/property_map/property_map.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/feature.h>
#include <pcl/common/centroid.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/pca.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_circle3D.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkPointData.h> 
#include <vtkCellArray.h>
#include <vtkCellData.h> 
#include <vtkDoubleArray.h>
#include <vtkTriangle.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>

#include <easy3d/core/types.h>
#include <easy3d/fileio/point_cloud_io.h>
#include <easy3d/core/point_cloud.h>


/*
 * This file incorporates and modifies portions of code from an open-source project licensed under the GNU General Public License (GPL) v3 .
 * 
 * Original Code Author: [DU Shenglan]
 * Source: [https://github.com/tudelft3d/AdTree/tree/main]
 * License: GNU General Public License v3 
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * A copy of the license can be found in the LICENSE file included with this program.
 */

// extract wood
std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcl_build_kdtree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float radius);
Eigen::Vector3f calculateEigenvaluesPCA(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cluster);
pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsByIndices(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices);
std::vector<std::vector<int>> slicePointCloudByZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float slice_thickness);
bool dbscan(const pcl::PointCloud<pcl::PointXYZ>& cloud_in, std::vector<pcl::Indices>& cluster_idx, const double& epsilon, const int& minpts);
bool detectCircleFeature(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float distanceThreshold, int maxIterations, float f1, float f2);
void savePointCloudToTxt(const std::string& filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

// skeleton
class Skeleton
{
	typedef boost::adjacency_list<boost::setS, boost::vecS, boost::undirectedS, SGraphVertexProp, SGraphEdgeProp > Graph;
	typedef boost::graph_traits<Graph>::vertex_descriptor SGraphVertexDescriptor;
	typedef boost::graph_traits<Graph>::edge_descriptor SGraphEdgeDescriptor;
	typedef boost::graph_traits<Graph>::vertex_iterator SGraphVertexIterator;
	typedef boost::graph_traits<Graph>::edge_iterator SGraphEdgeIterator;
	typedef boost::graph_traits<Graph>::adjacency_iterator SGraphAdjacencyIterator;
	typedef boost::graph_traits<Graph>::out_edge_iterator  SGraphOutEdgeIterator;
	
	struct Branch {
        std::vector<easy3d::vec3> points;
        std::vector<double>       radii;
		std::vector<int> level;
		int branchLevel;
    };
	
	std::vector<std::pair<int, double>> getTreeVolume() const;
	std::vector<std::tuple<int, double, double, double>> getBranchAttribute() const;
	const Graph& get_skeleton() const { return smoothed_skeleton_; }
	
	bool buildMST(const easy3d::PointCloud* cloud);
	bool buildTrunkMST();
	Graph mergedGraphs(const Graph& graph1, const Graph& graph2);
	bool extract_maintrunkSke(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void easy3d2pcl(const std::vector<easy3d::vec3>& point3d, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
	void keep_main_skeleton(Graph *i_Graph, double subtree_Threshold);
	void merge_collapsed_edges();
	bool skeletonSimplified();
	void get_graph_for_smooth(std::vector<Path> &pathList);
	bool skeletonSmooth();
	void calculateBranchAttribute(const std::vector<Branch> branches);
	std::vector<std::tuple<int, int>> constructMinimumSpanningTree(const std::vector<easy3d::vec3>& points);
	void buildGraph(Graph& graph, const std::vector<easy3d::vec3>& points, const std::vector<double>& radii);
	void saveGraphToPLY(const Graph& graph, const std::string& filename)；
	
	
	
private:
	
	Vector3D* Points_;
	KdTree* KDtree_;
	std::vector<Vector3D> mainTrunkSk_;
	std::vector<Vector3D> trunkPoints_;

	
    
    Graph   MST_;
	Graph   TrunkMST_;
    Graph   simplified_skeleton_;
    Graph   smoothed_skeleton_;
	int finalLevel_;

	

	
	SGraphVertexDescriptor RootV_;
	Vector3D RootPos_;
	double TrunkRadius_;
	double TreeHeight_;
	double BoundingDistance_;
	Branch branchTrunk;

	
	
	
	std::vector<std::pair<int, double>> treeVolume;
	
	std::vector<std::tuple<int, double, double, double>> branchAttribute;

    bool   quiet_;
}



