#include "skeleton.h"

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> pcl_build_kdtree(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, float radius)
{
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_clusters;
	std::vector<bool> processed(cloud_in->points.size(), false);

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud_in);
	srand((int)time(0));
	for (size_t i = 0; i < cloud_in->points.size(); ++i)
	{
		if (processed[i])
		{
			continue;
		}
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

		std::vector<int> point_indices;
		std::vector<float> point_diatances;

		if (kdtree.radiusSearch(cloud_in->points[i], radius, point_indices, point_diatances) > 0)
		{
			uint8_t r = rand() % (256) + 0;
			uint8_t g = rand() % (256) + 0;
			uint8_t b = rand() % (256) + 0;
			for (size_t j = 0; j < point_indices.size(); ++j)
			{
				int index = point_indices[j];
				if (!processed[index])
				{
					pcl::PointXYZRGB temp;
					temp.x = cloud_in->points[index].x;
					temp.y = cloud_in->points[index].y;
					temp.z = cloud_in->points[index].z;
					temp.r = r;
					temp.g = g;
					temp.b = b;
					cluster->points.push_back(temp);
					processed[index] = true;
				}
			}
			cluster->width = cluster->points.size();
			cluster->height = 1;
			cluster->is_dense = true;
			cloud_clusters.push_back(cluster);
		}
	}
	return cloud_clusters;
}

Eigen::Vector3f calculateEigenvaluesPCA(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cluster)
{
	pcl::PCA<pcl::PointXYZRGB> pca;
	pca.setInputCloud(cluster);
	Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
	Eigen::Vector3f eigen_values = pca.getEigenValues();

	return eigen_values;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr extractPointsByIndices(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr extracted_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (int index : indices)
	{
		if (index >= 0 && index < cloud->size())
		{
			extracted_cloud->push_back(cloud->at(index));
		}
	}
	return extracted_cloud;
}



std::vector<std::vector<int>> Skeleton::slicePointCloudByZ(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float slice_thickness)
{
	std::vector<std::vector<int>> slice_indices;
	std::vector<int> sorted_indices(cloud->size());
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		sorted_indices[i] = static_cast<int>(i);
	}
	std::sort(sorted_indices.begin(), sorted_indices.end(),
		[cloud](int idx1, int idx2) {
			return cloud->points[idx1].z < cloud->points[idx2].z;
		});
	double current_z = cloud->points[sorted_indices[0]].z;
	slice_indices.emplace_back();
	slice_indices[0].push_back(sorted_indices[0]);
	for (size_t i = 1; i < cloud->size(); ++i)
	{
		double z = cloud->points[sorted_indices[i]].z;
		if (z - current_z > slice_thickness)
		{
			slice_indices.emplace_back();
			current_z = z;
		}
		slice_indices.back().push_back(sorted_indices[i]);
	}

	return slice_indices;
}

// dbscan
bool Skeleton::dbscan(const pcl::PointCloud<pcl::PointXYZ>& cloud_in,  std::vector<pcl::Indices>& cluster_idx, const double& epsilon, const int& minpts)
{
	
	std::vector<bool> cloud_processed(cloud_in.size(), false);

	for (size_t i = 0; i < cloud_in.size(); ++i)
	{
		if (cloud_processed[i] != false)
		{
			continue;
		}
		pcl::Indices seed_queue;
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
		tree->setInputCloud(cloud_in.makeShared());
		pcl::Indices k_indices;
		std::vector<float> k_distances;
		if (tree->radiusSearch(cloud_in.points[i], epsilon, k_indices, k_distances) >= minpts)
		{
			seed_queue.push_back(i);
			cloud_processed[i] = true;
		}
		else
		{
			continue;
		}

		int seed_index = 0;
		while (seed_index < seed_queue.size())
		{
			pcl::Indices indices;
			std::vector<float> dists;
			if (tree->radiusSearch(cloud_in.points[seed_queue[seed_index]], epsilon, indices, dists) < minpts)
			{
				
				++seed_index;
				continue;
			}
			for (size_t j = 0; j < indices.size(); ++j)
			{
				if (cloud_processed[indices[j]])
				{
					continue;
				}
				seed_queue.push_back(indices[j]);
				cloud_processed[indices[j]] = true;
			}
			++seed_index;
		}

		cluster_idx.push_back(seed_queue);

	}

	if (cluster_idx.size() == 0)
		return false;
	else
		return true;
}

// ransac 
bool detectCircleFeature(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float distanceThreshold, int maxIterations, float f1, float f2)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1 = cloud;
	pcl::PointXYZ centroid = computeCentroid(cloud1);
	pcl::SampleConsensusModelCircle3D<pcl::PointXYZRGB>::Ptr model_circle3D(new pcl::SampleConsensusModelCircle3D<pcl::PointXYZRGB>(cloud1));
	pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac(model_circle3D);
	ransac.setDistanceThreshold(distanceThreshold);
	ransac.setMaxIterations(maxIterations);
	ransac.computeModel();
	pcl::IndicesPtr inliers(new std::vector<int>());
	ransac.getInliers(*inliers);

	Eigen::VectorXf coeff;
	ransac.getModelCoefficients(coeff);
	
	if (inliers->size() < 50 ) 
	{
		return false;
	}
	else if (coeff[3] < 0 || coeff[3] > 10.0) 
	{
		return false;
	}
	else if (coeff[4] * coeff[4] + coeff[5] * coeff[5] + coeff[6] * coeff[6] < 0.9) 
	{
		return false;
	}
	else
	{
		return true;
	}
}

// save data to txt
void savePointCloudToTxt(const std::string& filename, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
	std::ofstream outfile(filename);
	if (outfile.is_open()) {
		for (const auto& point : cloud->points) {
			outfile << point.x << " " << point.y << " " << point.z << std::endl;
		}
		outfile.close();
	}
	else {
		std::cerr << "Unable to open file: " << filename << std::endl;
	}
}

// extract maintrunkske
bool Skeleton::extract_maintrunkSke(const easy3d::PointCloud* cloud)
{
	std::vector<easy3d::vec3> point3d = cloud->points();
	pcl::PointCloud<pcl::PointXYZ>::Ptr point(new pcl::PointCloud<pcl::PointXYZ>);
	easy3d2pcl(point3d, point);
	std::vector<std::vector<int>> slicePointClouds;
	slicePointClouds = slicePointCloudByZ(point, );
	pcl::PointCloud<pcl::PointXYZ>::Ptr trunk(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < slicePointClouds.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr extract_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		extract_cloud = extractPointsByIndices(point, slicePointClouds[i]);

		
		std::vector<pcl::Indices> dbscancluster_indices;
		dbscan(*extract_cloud, dbscancluster_indices, , );
		if (dbscancluster_indices.size() == 1)
		{
			*trunk += *extract_cloud;
			Vector3D centroid;
			centroid.x = centroid.y = centroid.z = 0;
			for (const auto& p : *extract_cloud) {	
				centroid.x += p.x;
				centroid.y += p.y;
				centroid.z += p.z;
			}
			centroid.x /= extract_cloud->size();
			centroid.y /= extract_cloud->size();
			centroid.z /= extract_cloud->size();
			mainTrunkSk_.push_back(centroid);
			
			pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			for (const auto& p : *extract_cloud) {
				pcl::PointXYZ projected_point;
				projected_point.x = p.x;
				projected_point.y = p.y;
				projected_point.z = 0.0;
				projected_cloud->push_back(projected_point);
			}

			pcl::PointXYZ min_pt, max_pt;
			pcl::getMinMax3D(*projected_cloud, min_pt, max_pt);
			double x_range = max_pt.x - min_pt.x;
			double y_range = max_pt.y - min_pt.y;
			double max_range = std::max(x_range, y_range);
			branchTrunk.points.push_back({ centroid.x,centroid.y,centroid.z });
			branchTrunk.radii.push_back(max_range/2);
			branchTrunk.level.push_back(0);
			branchTrunk.branchLevel = 0;
		}
	}
	for (const auto& p : *trunk)
	{
		trunkPoints_.push_back({ p.x,p.y,p.z });
	}
	
	TrunkRadius_ = trunkPoints[0].radii;

	
	return true;
}
// data trans
void Skeleton::easy3d2pcl(const std::vector<easy3d::vec3>& point3d, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	for (size_t i = 0; i < point3d.size(); i++)
	{
		pcl::PointXYZ point;
		point.x = point3d[i].x;
		point.y = point3d[i].y;
		point.z = point3d[i].z;
		cloud->points.push_back(point);
	}
}

bool Skeleton::buildMST(const PointCloud* cloud)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr c(new pcl::PointCloud<pcl::PointXYZ>);
	easy3d2pcl(cloud,c)
	extract_maintrunkSke(c);
	buildTrunkMST();
	const std::vector<easy3d::vec3>& points = cloud->vec3;
	std::vector<std::tuple<int, int>> edges = constructMinimumSpanningTree(const std::vector<easy3d::vec3>& points);
	Graph& graph;
	buildGraph(graph,edges,const std::vector<double>& radii);
	mergedGraphs(TrunkMST_,graph);
	
	return true;
}

std::vector<std::pair<int, double>> Skeleton::getTreeVolume() const
{
	return treeVolume;
}

std::vector<std::tuple<int, double, double, double>> Skeleton::getBranchAttribute() const
{
	return branchAttribute;
}

bool Skeleton::buildTrunkMST()
{

	TrunkMST_.clear();

	std::vector<SGraphVertexDescriptor> vp;
	for (const auto& v : mainTrunkSk_)
	{
		SGraphVertexProp vertexProp;
		vertexProp.cVert = easy3d::vec3(v.x, v.y, v.z);
		vertexProp.nParent = 0;
		vertexProp.lengthOfSubtree = 0.0;
		vp.push_back(add_vertex(vertexProp, TrunkMST_));
	}

	std::vector<std::pair<SGraphVertexDescriptor, double>> sortedVertices;
	for (std::size_t i = 0; i < num_vertices(TrunkMST_); i++)
	{
		sortedVertices.emplace_back(vertex(i, TrunkMST_), get(boost::vertex_bundle, TrunkMST_, vertex(i, TrunkMST_)).cVert.z);
	}
	std::sort(sortedVertices.begin(), sortedVertices.end(), [](const auto& a, const auto& b) { return a.second < b.second; });

	SGraphVertexDescriptor prevVertex = sortedVertices[0].first;
	for (std::size_t i = 1; i < sortedVertices.size(); i++)
	{
		SGraphVertexDescriptor currVertex = sortedVertices[i].first;
		double distance = easy3d::distance(get(boost::vertex_bundle, TrunkMST_, prevVertex).cVert,
			get(boost::vertex_bundle, TrunkMST_, currVertex).cVert);
		SGraphEdgeProp edgeProp;
		edgeProp.nWeight = distance;
		edgeProp.nRadius = 0.0;
		edgeProp.vecPoints.clear();
		boost::add_edge(prevVertex, currVertex, edgeProp, TrunkMST_);
		get(boost::vertex_bundle, TrunkMST_, currVertex).nParent = prevVertex;
		prevVertex = currVertex;
	}

	return true;
}

Graph Skeleton::mergedGraphs(const Graph& graph1, const Graph& graph2)
{
	
	double maxZ2 = -std::numeric_limits<double>::max();
	for (auto vp = vertices(graph2); vp.first != vp.second; ++vp.first) {
		SGraphVertexDescriptor v = *vp.first;
		if (graph2[v].cVert.z > maxZ2) {
			maxZ2 = graph2[v].cVert.z;
		}
	}

	
	Graph mergedGraph;

	
	std::map<SGraphVertexDescriptor, SGraphVertexDescriptor> vertexMap2;
	for (auto vp = vertices(graph2); vp.first != vp.second; ++vp.first) {
		SGraphVertexDescriptor v = *vp.first;
		SGraphVertexDescriptor new_v = add_vertex(graph2[v], mergedGraph);
		vertexMap2[v] = new_v;
	}
	for (auto ep = edges(graph2); ep.first != ep.second; ++ep.first) {
		SGraphEdgeDescriptor e = *ep.first;
		add_edge(vertexMap2[source(e, graph2)], vertexMap2[target(e, graph2)], graph2[e], mergedGraph);
	}

	
	std::map<SGraphVertexDescriptor, SGraphVertexDescriptor> vertexMap1;
	for (auto vp = vertices(graph1); vp.first != vp.second; ++vp.first) {
		SGraphVertexDescriptor v = *vp.first;
		if (graph1[v].cVert.z > maxZ2) {
			SGraphVertexDescriptor new_v = add_vertex(graph1[v], mergedGraph);
			vertexMap1[v] = new_v;
		}
	}
	for (auto ep = edges(graph1); ep.first != ep.second; ++ep.first) {
		SGraphEdgeDescriptor e = *ep.first;
		SGraphVertexDescriptor src = source(e, graph1);
		SGraphVertexDescriptor tgt = target(e, graph1);
		if (vertexMap1.find(src) != vertexMap1.end() && vertexMap1.find(tgt) != vertexMap1.end()) {
			add_edge(vertexMap1[src], vertexMap1[tgt], graph1[e], mergedGraph);
		}
	}

	
	SGraphVertexDescriptor minVertex1;
	double minZ1 = std::numeric_limits<double>::max();
	for (auto& pair : vertexMap1) {
		SGraphVertexDescriptor v = pair.first;
		if (graph1[v].cVert.z < minZ1) {
			minZ1 = graph1[v].cVert.z;
			minVertex1 = v;
		}
	}

	SGraphVertexDescriptor maxVertex2;
	for (auto& pair : vertexMap2) {
		SGraphVertexDescriptor v = pair.first;
		if (graph2[v].cVert.z == maxZ2) {
			maxVertex2 = v;
			break;
		}
	}

	
	SGraphEdgeProp edgeProp;
	edgeProp.nWeight = 1.0; 
	edgeProp.nRadius = 1.0; 
	add_edge(vertexMap1[minVertex1], vertexMap2[maxVertex2], edgeProp, mergedGraph);

	return mergedGraph;
}

bool Skeleton::skeletonSimplified()
{
    if (!quiet_)
        std::cout << "step 1: eliminate unimportant small edges" << std::endl;
    keep_main_skeleton(&MST_, );

    if (!quiet_)
        std::cout << "step 2: iteratively merge collapsed edges" << std::endl;
    merge_collapsed_edges();

    if (!quiet_)
        std::cout << "finish the skeleton graph refining!" << std::endl;
	
	computeSubtreeLength(&simplified_skeleton_, RootV_);
    computeWeight(&simplified_skeleton_);
	
	return true;
}

void Skeleton::keep_main_skeleton(Graph *i_Graph, double subtree_Threshold)
{
    simplified_skeleton_.clear();

	std::pair<SGraphVertexIterator, SGraphVertexIterator> vp = boost::vertices(*i_Graph);

	for (SGraphVertexIterator cIter = vp.first; cIter != vp.second; ++cIter)
	{
		SGraphVertexProp pV;
		pV.cVert = (*i_Graph)[*cIter].cVert;
		pV.nParent = (*i_Graph)[*cIter].nParent;
		pV.lengthOfSubtree = (*i_Graph)[*cIter].lengthOfSubtree;
        add_vertex(pV, simplified_skeleton_);
	}

	
	std::vector<SGraphVertexDescriptor> stack;
	stack.push_back(RootV_);
	while (true)
	{
		SGraphVertexDescriptor currentV = stack.back();
		stack.pop_back();
		std::pair<SGraphAdjacencyIterator, SGraphAdjacencyIterator> aj = adjacent_vertices(currentV, *i_Graph);
		
		for (SGraphAdjacencyIterator aIter = aj.first; aIter != aj.second; ++aIter)
		{
			
			if (*aIter != (*i_Graph)[currentV].nParent)
			{
				double child2Current = std::sqrt((*i_Graph)[currentV].cVert.distance2((*i_Graph)[*aIter].cVert));
				double subtreeRatio = ((*i_Graph)[*aIter].lengthOfSubtree + child2Current) / (*i_Graph)[currentV].lengthOfSubtree;
				if (subtreeRatio >= subtree_Threshold)
				{
					SGraphEdgeProp pEdge;
					SGraphEdgeDescriptor sEdge = edge(*aIter, currentV, (*i_Graph)).first;
					pEdge.nWeight = (*i_Graph)[sEdge].nWeight;
					pEdge.nRadius = (*i_Graph)[sEdge].nRadius;
					pEdge.vecPoints = (*i_Graph)[sEdge].vecPoints;
					SGraphVertexDescriptor dSource = source(sEdge, *i_Graph);
					SGraphVertexDescriptor dTarget = target(sEdge, *i_Graph);
                    add_edge(dSource, dTarget, pEdge, simplified_skeleton_);
					stack.push_back(*aIter);
				}
			}
		}
		if (stack.size() == 0)
			break;
	}
	
	computeSubtreeLength(&simplified_skeleton_, RootV_);
    computeWeight(&simplified_skeleton_);
    
	
	return;
}

// 
void Skeleton::merge_collapsed_edges()
{
	
    std::pair<SGraphVertexIterator, SGraphVertexIterator> vp = boost::vertices(simplified_skeleton_);
	bool bChange = true;
	int numComplex = 0;
	while (bChange)
	{
		bChange = false;
		for (SGraphVertexIterator cIter = vp.first; cIter != vp.second; ++cIter)
		{
			SGraphVertexDescriptor dVertex = *cIter;
			
           
            else if (simplified_skeleton_[dVertex].nParent != dVertex)
			{
                if (check_single_child_vertex(&simplified_skeleton_, dVertex))
				{
					bChange = true;
					numComplex++;
				}
			}
		}
	}

    compute_length_of_subtree(&simplified_skeleton_, RootV_);
    compute_graph_edges_weight(&simplified_skeleton_);
    compute_all_edges_radius(TrunkRadius_);

	return;
}

void Skeleton::get_graph_for_smooth(std::vector<Path> &pathList)
{
	pathList.clear();
	Path currentPath;
	int cursor = 0;
	
	currentPath.push_back(RootV_);
	pathList.push_back(currentPath);
	
	while (cursor < pathList.size())
	{
		currentPath = pathList[cursor];
		
		SGraphVertexDescriptor endV = currentPath.back();
		
        if ((out_degree(endV, simplified_skeleton_) == 1) && (endV != simplified_skeleton_[endV].nParent))
			cursor++;
		else
		{
			
			double maxR = -1;
			int isUsed = -1;
			SGraphVertexDescriptor fatestChild;
			std::vector<SGraphVertexDescriptor> notFastestChildren;
            std::pair<SGraphAdjacencyIterator, SGraphAdjacencyIterator> adjacencies = adjacent_vertices(endV, simplified_skeleton_);
			for (SGraphAdjacencyIterator cIter = adjacencies.first; cIter != adjacencies.second; ++cIter)
			{
                if (*cIter != simplified_skeleton_[endV].nParent)
				{
                    SGraphEdgeDescriptor currentE = edge(endV, *cIter, simplified_skeleton_).first;
                    double radius = simplified_skeleton_[currentE].nRadius;
					if (maxR < radius)
					{
						maxR = radius;
						if (isUsed > -1)
							notFastestChildren.push_back(fatestChild);
						else
							isUsed = 0;
						fatestChild = *cIter;
					}
					else
						notFastestChildren.push_back(*cIter);
				}
			}
			
			for (int nChild = 0; nChild < notFastestChildren.size(); ++nChild)
			{
				Path newPath;
				newPath.push_back(endV);
				newPath.push_back(notFastestChildren[nChild]);
				pathList.push_back(newPath);
			}
			
			pathList[cursor].push_back(fatestChild);
		}
	}
	
	return;
}

// smooth mst
bool Skeleton::skeletonSmooth()
{
    if (num_edges(simplified_skeleton_) < 2) {
        std::cout << "skeleton does not exist!" << std::endl;
        return false;
    }

    smoothed_skeleton_.clear();
    if (!quiet_)
        std::cout << "smoothing skeleton..." << std::endl;

   
    std::vector<Path> pathList;
    get_graph_for_smooth(pathList);

    
    for (std::size_t n_path = 0; n_path < pathList.size(); ++n_path)
    {
        Path currentPath = pathList[n_path];
        std::vector<vec3> interpolatedPoints;
        std::vector<double> interpolatedRadii;
        static int numOfSlices = 20;
        std::vector<int> numOfSlicesCurrent;
        
        for (std::size_t n_node = 0; n_node < currentPath.size() - 1; ++n_node)
        {
            SGraphVertexDescriptor sourceV = currentPath[n_node];
            SGraphVertexDescriptor targetV = currentPath[n_node + 1];
            vec3 pSource = simplified_skeleton_[sourceV].cVert;
            vec3 pTarget = simplified_skeleton_[targetV].cVert;
            float branchlength = easy3d::distance(pSource, pTarget);
            numOfSlicesCurrent.push_back(std::max(static_cast<int>(branchlength * numOfSlices), 2));

           
            vec3 tangentOfSorce;
            vec3 tangentOfTarget;
            
            if (sourceV == simplified_skeleton_[sourceV].nParent)
                tangentOfSorce = (pTarget - pSource).normalize();
            else
            {
                SGraphVertexDescriptor parentOfSource = simplified_skeleton_[sourceV].nParent;
                tangentOfSorce = (pTarget - simplified_skeleton_[parentOfSource].cVert).normalize();
            }
            
            if ((out_degree(targetV, simplified_skeleton_) == 1) && (targetV != simplified_skeleton_[targetV].nParent))
                tangentOfTarget = (pTarget - pSource).normalize();
            else
            {
                SGraphVertexDescriptor childOfTarget = currentPath[n_node + 2];
                tangentOfTarget = (simplified_skeleton_[childOfTarget].cVert - pSource).normalize();
            }

            tangentOfSorce *= branchlength;
            tangentOfTarget *= branchlength;

            
            vec3 A = tangentOfTarget + tangentOfSorce + 2 * (pSource - pTarget);
            vec3 B = 3 * (pTarget - pSource) - 2 * tangentOfSorce - tangentOfTarget;
            vec3 C = tangentOfSorce;
            vec3 D = pSource;
            SGraphEdgeDescriptor currentE = edge(sourceV, targetV, simplified_skeleton_).first;
            double sourceRadius = simplified_skeleton_[currentE].nRadius;
            double targetRadius = sourceRadius;
            SGraphVertexDescriptor ParentVert = simplified_skeleton_[sourceV].nParent;
            if (ParentVert != sourceV)
            {
                SGraphEdgeDescriptor ParentEdge = edge(ParentVert, sourceV, simplified_skeleton_).first;
                sourceRadius = simplified_skeleton_[ParentEdge].nRadius;
            }
            double deltaOfRadius = (sourceRadius - targetRadius) / numOfSlicesCurrent[numOfSlicesCurrent.size() - 1];
           
            for (std::size_t n = 0; n < numOfSlicesCurrent[numOfSlicesCurrent.size() - 1]; ++n)
            {
                double t = static_cast<double>(static_cast<double>(n) / numOfSlicesCurrent[numOfSlicesCurrent.size() - 1]);
                vec3 point = A * t*t*t + B * t*t + C * t + D;

                if (n == 0) {
                    interpolatedPoints.push_back(point);
                    interpolatedRadii.push_back(sourceRadius - n * deltaOfRadius);
                }
                else {
                    const vec3& prev = interpolatedPoints.back();
                    if (distance2(prev, point) > epsilon<float>() * 10) { 
                        interpolatedPoints.push_back(point);
                        interpolatedRadii.push_back(sourceRadius - n * deltaOfRadius);
                    }
                }
            }
        }
        
        SGraphVertexDescriptor endV = currentPath.back();
        const vec3& prev = interpolatedPoints.back();
        const vec3& point = simplified_skeleton_[endV].cVert;
        if (distance2(prev, point) > epsilon<float>() * 10) { 
            interpolatedPoints.push_back(point);
            interpolatedRadii.push_back(0);
        }

		if (interpolatedPoints.size() < 2)
			continue; 

        
        std::vector<SGraphVertexDescriptor> vertices;
        for (std::size_t np = 0; np < interpolatedPoints.size(); np++) {
            SGraphVertexProp vp;
            vp.cVert = interpolatedPoints[np];
            vp.radius = interpolatedRadii[np];
            SGraphVertexDescriptor v = add_vertex(vp, smoothed_skeleton_);
            vertices.push_back(v);
        }

       
        for (std::size_t np = 0; np < vertices.size() - 1; np++) {
            add_edge(vertices[np], vertices[np + 1], SGraphEdgeProp(), smoothed_skeleton_);
        }
    }
	
    return true;
}

// build mst
std::vector<std::tuple<int, int>> Skeleton::constructMinimumSpanningTree(const std::vector<easy3d::vec3>& points)
{
	int n = points.size();
	Graph g(n);

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			float weight = (points[i] - points[j]).norm();
			add_edge(i, j, weight, g);
		}
	}

	std::vector<Vertex> p(num_vertices(g));
	Vertex start_vertex = std::min_element(points.begin(), points.end(), [](const easy3d::vec3& a, const easy3d::vec3& b) { return a.z < b.z; }) - points.begin();
	prim_minimum_spanning_tree(g, &p[0], root_vertex(start_vertex));

	std::vector<std::tuple<int, int>> skeletonEdges;
	for (std::size_t i = 0; i != p.size(); ++i) {
		if (p[i] != i) {
			skeletonEdges.emplace_back(p[i], i);
		}
	}

	return skeletonEdges;
}
void Skeleton::buildGraph(Graph& graph, const std::vector<easy3d::vec3>& points, const std::vector<double>& radii)
{
	for (size_t i = 0; i < points.size(); ++i) {
		for (size_t j = 0; j < points.size(); ++j) {
			double weight = (points[i] - points[j]).norm();
			boost::add_edge(i, j, SGraphEdgeProp{ weight,0.0,{},0 }, graph);
		}
	}
}


// BranchAttribute
void Skeleton::calculateBranchAttribute(const std::vector<Branch> branches)
{
	for (int i = 1; i < branches.size(); i++)
	{
		double branchLength = 0.0;
		double bcl = 0.0;
		if (branches[i].branchLevel == 1) {
			
			bcl = (branches[i].points.back() - branches[i].points.front()).length();
			for (int j = 0; j < branches[i].points.size()-1; j++) {
				double length = (branches[i].points[j + 1] - branches[i].points[j]).length();
				branchLength += length;
			}
			int sor = findPointIndex(branches[0], branches[i].points[0]);
			const vec3 a = branches[i].points[1];
			const vec3 b = branches[i].points[0];
			const vec3 c = branches[0].points[sor + 1];

			vec3 ab = a - b;
			vec3 bc = b - c;  

			
			double dot_product = ab.x * bc.x + ab.y * bc.y + ab.z * bc.z;
			vec3 cross_product = vec3(
				ab.y * bc.z - ab.z * bc.y,
				ab.z * bc.x - ab.x * bc.z,
				ab.x * bc.y - ab.y * bc.x
			);
			double cross_product_length = cross_product.length();

			
			double angle_rad = std::atan2(cross_product_length, dot_product);

			
			double angle_deg = angle_rad * 180.0 / M_PI;


			branchAttribute.push_back(std::make_tuple(i, angle_deg, branchLength, bcl));
		}
	}
}

void Skeleton::saveGraphToPLY(const Graph& graph, const std::string& filename) 
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open file: " << filename << std::endl;
		return;
	}

	for (auto vp = vertices(graph); vp.first != vp.second; ++vp.first) {
		SGraphVertexDescriptor v = *vp.first;
		file << "v " << graph[v].cVert.x << " " << graph[v].cVert.y << " " << graph[v].cVert.z << std::endl;
	}

	for (auto ep = edges(graph); ep.first != ep.second; ++ep.first) {
		SGraphEdgeDescriptor e = *ep.first;
		SGraphVertexDescriptor src = source(e, graph);
		SGraphVertexDescriptor tgt = target(e, graph);
		file << "l " << (src + 1) << " " << (tgt + 1) << std::endl;
	}

	file.close();
}



int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("", *cloud);
	

	pcl::PointCloud<pcl::PointXYZ>::Ptr maintrunkPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr xixiaoTrunkPoint(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr leafPointcloud(new pcl::PointCloud<pcl::PointXYZ>);


	
	std::vector<std::vector<int>> slicePointClouds;
	slicePointClouds = slicePointCloudByZ(cloud, 0.1);
	

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	for (int i = 0; i < slicePointClouds.size(); ++i)
	{
		
		uint8_t R1 = rand() % (256) + 0;
		uint8_t G1 = rand() % (256) + 0;
		uint8_t B1 = rand() % (256) + 0;
		pcl::PointCloud<pcl::PointXYZ>::Ptr extract_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		extract_cloud = extractPointsByIndices(cloud, slicePointClouds[i]);
		for (size_t j = 0; j < extract_cloud->points.size(); j++)
		{
			pcl::PointXYZRGB p;
			p.x = extract_cloud->points[j].x;
			p.y = extract_cloud->points[j].y;
			p.z = extract_cloud->points[j].z;
			p.r = R1;
			p.g = G1;
			p.b = B1;
			pointcloud->points.push_back(p);
		}
		
		std::vector<pcl::Indices> dbscancluster_incides;
		dbscan(*extract_cloud, dbscancluster_incides, x, x);

		
		std::vector<bool> is_noise(extract_cloud->points.size(), true);

		if (dbscancluster_incides.size() == 1)
		{
			*maintrunkPoint += *extract_cloud;
			
			std::fill(is_noise.begin(), is_noise.end(), false);
		}
		else
		{
			int begin = 0;
			for (std::vector<pcl::Indices>::const_iterator it = dbscancluster_incides.begin(); it != dbscancluster_incides.end(); ++it)
			{
				
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_dbscan(new pcl::PointCloud<pcl::PointXYZRGB>);
				uint8_t R = rand() % (256) + 0;
				uint8_t G = rand() % (256) + 0;
				uint8_t B = rand() % (256) + 0;
				for (auto pit = it->begin(); pit != it->end(); ++pit)
				{
					pcl::PointXYZRGB point_db;
					point_db.x = extract_cloud->points[*pit].x;
					point_db.y = extract_cloud->points[*pit].y;
					point_db.z = extract_cloud->points[*pit].z;
					point_db.r = R;
					point_db.g = G;
					point_db.b = B;
					cloud_dbscan->points.push_back(point_db);

					
					is_noise[*pit] = false;
				}

				if (cloud_dbscan->points.size() >= 300)
				{
					bool isCiecleFeature = detectCircleFeature(cloud_dbscan, 0.1, 500, 1, 1);
					if (!isCiecleFeature)
					{
						return;
					}	
					else
					{
						for (const auto& pp : cloud_dbscan->points)
						{
							pcl::PointXYZ p1;
							p1.x = pp.x;
							p1.y = pp.y;
							p1.z = pp.z;
							maintrunkPoint->points.push_back(p1);
						}
					}
				}
				else
				{
					Eigen::Vector3f pcaEigen = calculateEigenvaluesPCA(cloud_dbscan);

					float L = (pcaEigen[0] - pcaEigen[1]) / pcaEigen[1];
					if (L > 0.9)
					{
						for (const auto& pp : cloud_dbscan->points)
						{
							pcl::PointXYZ pp1;
							pp1.x = pp.x;
							pp1.y = pp.y;
							pp1.z = pp.z;
							xixiaoTrunkPoint->points.push_back(pp1);
						}
					}
					else
					{
						for (const auto& pp : cloud_dbscan->points)
						{
							pcl::PointXYZ pp1;
							pp1.x = pp.x;
							pp1.y = pp.y;
							pp1.z = pp.z;
							leafPointcloud->points.push_back(pp1);
						}
					}
				}
				
		    }
		
		}
		
		for (size_t j = 0; j < extract_cloud->points.size(); j++)
		{
			if (is_noise[j])
			{
				pcl::PointXYZ pp1;
				pp1.x = extract_cloud->points[j].x;
				pp1.y = extract_cloud->points[j].y;
				pp1.z = extract_cloud->points[j].z;
				leafPointcloud->points.push_back(pp1);
			}
		}

	}

	easy3d::PointCloud* cloud1 = easy3d::PointCloudIO::load("xxx.xyz");
	Skeleton sk;
	sk.buildMST(cloud1);
	
	sk.skeletonSimplified();
	sk.skeletonSmooth();
	
	Graph g = sk.get_skeleton();
	std::vector<std::pair<int,double>> treeVolume = sk.getTreeVolume();
	std::vector<std::tuple<int, double, double,double>> treeBranchAttribute = sk.getBranchAttribute();
	
	return 0;
}