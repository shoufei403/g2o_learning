#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class SimpleVertex : public g2o::BaseVertex<1, double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SimpleVertex(bool fixed = false)
  {
    setToOriginImpl();
    setFixed(fixed);
  }

  SimpleVertex(double e, bool fixed = false)
  {
    _estimate = e;
    setFixed(fixed);
  }

  // 重置
  virtual void setToOriginImpl() override {
    _estimate = 0;
  }

  // 更新
  virtual void oplusImpl(const double *update) override {
    _estimate += *update;
  }

  // 存盘和读盘：留空
  virtual bool read(istream &in) {return true;}

  virtual bool write(ostream &out) const {return true;}
};

// 误差模型 模板参数：误差值维度，测量值类型，连接顶点类型
class SimpleUnaryEdge : public g2o::BaseUnaryEdge<1, double, SimpleVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimpleUnaryEdge() : BaseUnaryEdge() {}

  // 计算模型误差
  virtual void computeError() override {
    const SimpleVertex *v = static_cast<const SimpleVertex *> (_vertices[0]);
    const double abc = v->estimate();
    _error(0, 0) = _measurement - abc;
  }

  // 计算雅可比矩阵
  virtual void linearizeOplus() override {
    //此处就是求误差想对于abc变量的导数
    _jacobianOplusXi[0] = -1;//一元边只有一个顶点，所以只有_jacobianOplusXi。_jacobianOplusXi的维度和顶点需优化的维度是一致的
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

};

// 误差模型 模板参数：误差值维度，测量值类型，连接顶点类型
class SimpleBinaryEdge : public g2o::BaseBinaryEdge<1, double, SimpleVertex, SimpleVertex> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SimpleBinaryEdge() {}

  // 计算模型误差
  virtual void computeError() override {
    const SimpleVertex *v1 = static_cast<const SimpleVertex *> (_vertices[0]);
    const SimpleVertex *v2 = static_cast<const SimpleVertex *> (_vertices[1]);
    const double abc1 = v1->estimate();
    const double abc2 = v2->estimate();
    _error[0] = _measurement - (abc1 - abc2);
  }

  // 计算雅可比矩阵
  virtual void linearizeOplus() override {
    //此处就是求误差相对于abc1变量的导数
    _jacobianOplusXi(0,0) = -1;//二元边有两个顶点，偏差_error对每个顶点的偏导数都需要求解;如果_error是多维的，则每一维的偏导都需要求解。即会出现_jacobianOplusXi(1,0)

    //此处就是求误差相对于abc2变量的导数
    _jacobianOplusXj(0,0) = 1;//因为误差_error只有一维，所以_jacobianOplusXi只有_jacobianOplusXi(0,0)项。此时_jacobianOplusXi[0]与_jacobianOplusXi(0,0)效果等价。
  }

  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

};

int main(int argc, char **argv) {

  // 构建图优化，先设定g2o
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<1, 1>> BlockSolverType;  // 每个误差项优化变量维度为1，误差值维度为1
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型

  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmGaussNewton(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // 往图中增加顶点
  std::vector<SimpleVertex *> vertexs;

  SimpleVertex *v = new SimpleVertex();
  v->setEstimate(0);
  v->setId(0);
  v->setFixed(true);
  optimizer.addVertex(v);
  vertexs.push_back(v);

  SimpleVertex *v1 = new SimpleVertex();
  v1->setEstimate(1);
  v1->setId(1);
  optimizer.addVertex(v1);
  vertexs.push_back(v1);

  SimpleVertex *v2 = new SimpleVertex();
  v2->setEstimate(0.1);
  v2->setId(2);
  optimizer.addVertex(v2);
  vertexs.push_back(v2);

  // 往图中增加边
  SimpleUnaryEdge *edge = new SimpleUnaryEdge();
  // edge->setId(i);
  edge->setVertex(0, vertexs[0]);                // 设置连接的顶点
  edge->setMeasurement(0);      // 观测数值
  edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge(edge);

  SimpleBinaryEdge * edge1 = new SimpleBinaryEdge();
  // edge->setId(i);//id 不设置似乎没有关系，如果设置需要每条边设置成不一样的
  edge1->setVertex(0, vertexs[1]);//这里设置的序号对应的顶点要和边的computeError函数里设定的顶点是一一对应的
  edge1->setVertex(1, vertexs[0]);                // 设置连接的顶点
  edge1->setMeasurement(1);      // 观测数值
  edge1->setInformation(Eigen::Matrix<double, 1, 1>::Identity()*1.0); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge(edge1);

  SimpleBinaryEdge * edge2 = new SimpleBinaryEdge();
  // edge->setId(i);
  edge2->setVertex(0, vertexs[2]);                // 设置连接的顶点
  edge2->setVertex(1, vertexs[1]);                // 设置连接的顶点
  edge2->setMeasurement(-0.8);      // 观测数值
  edge2->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge(edge2);

  SimpleBinaryEdge * edge3 = new SimpleBinaryEdge();
  // edge->setId(i);
  edge3->setVertex(0, vertexs[2]);                // 设置连接的顶点
  edge3->setVertex(1, vertexs[0]);                // 设置连接的顶点
  edge3->setMeasurement(0);      // 观测数值
  edge3->setInformation(Eigen::Matrix<double, 1, 1>::Identity()); // 信息矩阵：协方差矩阵之逆
  optimizer.addEdge(edge3);


  // 执行优化
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // 输出优化值
  cout << "estimated model: " << vertexs[0]->estimate() << " " 
    << vertexs[1]->estimate() << " " << vertexs[2]->estimate() << " " << endl;



  return 0;
}