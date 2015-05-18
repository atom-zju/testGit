/*
  @ Usage: plug in kinect sensor and run the application,
  @
  @
  @
 */

#include <iostream>
#include <XnOS.h>
#include <GL/glut.h>
#include <math.h>
#include <fstream>
#include <XnCppWrapper.h>
#include <XnUSB.h>
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <algorithm>
#include <math>


#include <vector>
using namespace cv;
using namespace xn;
using namespace Eigen;
using namespace std;
#define width 640
#define height 480
#define lamda 100.0  //the most matchable match in sift is equally important to lamda icp matches points
//#define focalLen 580.0    //focal length in pixel
typedef Matrix<Vector3f,Dynamic,Dynamic>  kntMap;



//---------------------------------------------------------------------------
/////////////////////////  Define  //////////////////////////////////////////
//---------------------------------------------------------------------------

#define SAMPLE_XML_PATH "/home/atom/KinectLibs/OpenNI/Platform/Linux/Redist/OpenNI-Bin-Dev-Linux-x86-v1.5.7.10/Samples/Config/SamplesConfig.xml"

#define GL_WIN_SIZE_X 1280
#define GL_WIN_SIZE_Y 1024

#define DISPLAY_MODE_OVERLAY    1
#define DISPLAY_MODE_DEPTH        2
#define DISPLAY_MODE_IMAGE        3
#define DEFAULT_DISPLAY_MODE    DISPLAY_MODE_DEPTH

#define MAX_DEPTH 10000
#define PI 3.141592653
#define SIGN(x) ( (x)<0 ? -1:((x)>0?1:0 ) )

//-------------------------------uint--------------------------------------------
/////////////////////////  Globals  //////////////////////////////////////////
//---------------------------------------------------------------------------

//sift related globals
//        Point2i test;

//indicate whether to use sift match to merge two frames
int siftMerge=0;

Mat frame1=Mat(480,640,CV_8UC3,Scalar(0,0,0));
Mat frame2=Mat(480,640,CV_8UC3,Scalar(0,0,0));

int minHessian = 400;
SurfFeatureDetector detector( minHessian );
//SiftFeatureDetector detector(0.06f,10.0);
SurfDescriptorExtractor extractor;
//SiftDescriptorExtractor extractor;

vector<KeyPoint> keypoint1,keypoint2;
Mat descriptor1,descriptor2;

//BruteForceMatcher<L2<float> > matcher;
FlannBasedMatcher matcher;
vector<DMatch> matches;
vector< DMatch > good_matches;
Mat img_matches;
double max_dist = 0; double min_dist = 100;

//icp related globals
    kntMap range1(480,640),range2(480,640),range2Merged(480,640);
    kntMap normal1(480,640),normal2(480,640),normal2Merged(480,640);
    kntMap range1Low(240,320),range2Low(240,320);
    kntMap normal1Low(240,320),normal2Low(240,320);
    Matrix3f r0=MatrixXf::Identity(3,3);
    Vector3f t0=MatrixXf::Zero(3,1);
    MatrixXf rt;
    long regionCnt=53760,disValCnt=26880,norValCnt=13440;
    double e0upper=1.4,e0lower=1,edupper=0.03,edlower=0.0003;
    double augFac=1.0;

int focalLen=580;

unsigned int nNumberOfPoints = 0;               // number of valid points in a range image
float avgx,avgy,avgz;                           //avege x y z, in a rageimage;
float sumx=0,sumy=0,sumz=0;
float g_pDepthHist[MAX_DEPTH];                  //depth histogram

float thres=0.01;

XnRGB24Pixel* g_pTexMap = NULL;

int glWinWidth = 640, glWinHeight = 480;        //inital windowsize

double foucsx=0,foucsy=1,foucsz=0,angx=0,angy=0;//the point that you look at
static float angleX=0,angleY=0;

float view_radius=12;

unsigned int g_nTexMapX = 0;
unsigned int g_nTexMapY = 0;

unsigned int g_nViewState = DEFAULT_DISPLAY_MODE;


XnStatus rc;

double eyex=0, eyey=0, eyez=0, atx, aty, atz; // eye* - 摄像机位置，at* - 注视点位置
bool leftClickHold = false, rightClickHold = false;
int mx,my; // 鼠标按键时在 OpenGL 窗口的坐标
int ry = 90, rx = 90; // 摄像机相对注视点的观察角度
double mindepth=INFINITY, maxdepth=-1; // 深度数据的极值
double radius = 6000.0; // 摄像机与注视点的距离


uint texture[480][640][3];     //stores the global coordinate data
float xyzdata[480][640][3];

//stores the 2nd global coordinate data
uint texture2[480][640][3];
float xyzdata2[480][640][3];

//indicate whether the 2nd frame is valid
int mergeFlag=0;
//indicate whether the 1st frame is to be drawn
int primeFlag=1;
//indicate whether to use low definition for icp
int lowDefi=0;

//rotate matrix and translate matrix
double rotMat[3][3];
//double tranMat[3]={-0.8660254,0,0.5};
//double beta=-60.0/180*PI;        //press 'R' and rotate in clock, do not forget to add an addtional 0 in your para
double tranMat[3]={0,0,0};
double beta=0.0/180*PI;        //press 'R' and rotate in clock, do not forget to add an addtional 0 in your para

//int width=640, height=480;

//0:depth map 1:global model
int displayMode=0;

//0:do not use icp adjust 1:use icp adjust
int icpAdjust=0;

Context g_context;
ScriptNode g_scriptNode;
DepthGenerator g_depth;
ImageGenerator g_image;
DepthMetaData g_depthMD;
ImageMetaData g_imageMD;

std::ofstream fout;

bool distance_comparator(const DMatch& m1, const DMatch& m2){
    return m1.distance<m2.distance;
}

//---------------------------------------------------------------------------
/////////////////////////  ICP Func  ///////////////////////////////////////
//---------------------------------------------------------------------------

/********************  getNormal   *****************/

void getNormal(kntMap &range,kntMap &normal)
{

    Vector3f cross1,cross2,tmp;
    cout<<"begin calcu normal"<<endl;
    for(int i=0;i<range.rows()-1;i++){
        for(int j=0;j<range.cols()-1;j++){
            if(range(i,j)(2)!=INFINITY && range(i+1,j)(2)!=INFINITY && range(i,j+1)(2)!=INFINITY){
                cross1=range(i+1,j)-range(i,j);
                cross2=range(i,j+1)-range(i,j);
                tmp=cross1.cross(cross2);
                normal(i,j)=tmp/tmp.squaredNorm();
                //cout<<normal(i,j)<<endl<<endl;
            }
        }
    }
    //std::cout<<normal(0,0)<<std::endl;
}

/********************  myICP   *****************/

MatrixXf myICP(kntMap &range1,kntMap &normal1,kntMap &range2,kntMap &normal2, Matrix3f r0,Vector3f t0,double ed, double e0)
{
    int m=range1.rows();
    int n=range1.cols();
    cout<<"ed:"<<ed<<endl<<"e0:"<<e0<<endl;
    Matrix<float,6,6> sumA=MatrixXf::Zero(6,6);
    Matrix<float,6,1> sumATb=MatrixXf::Zero(6,1);
    Matrix<float,6,1> AT=MatrixXf::Zero(6,1);
    Matrix<float,6,1> x=MatrixXf::Zero(6,1);
    Vector3f tmp,tmpR;
    Matrix<float,3,6> G=MatrixXf::Zero(3,6);
    Matrix<float,3,4> res=MatrixXf::Ones(3,4);
    float b;
    G(0,3)=1;
    G(1,4)=1;
    G(2,5)=1;
    int u,v;
    long cnt=0;
    regionCnt=0;
    disValCnt=0;
    norValCnt=0;
    if(good_matches.size()<=5){
        cout<<"Total number of match points is less than 5, forced to use icp"<<endl;
        siftMerge=0;
        cout<<"use icp for frame merge"<<endl;
    }

    //if siftMerge==0 use icp for frame merge
    if(siftMerge==0){
        for(int i=0;i<m-1;i++){
            for(int j=0;j<n-1;j++){

                if(range2(i,j)(2)!=INFINITY && normal2(i,j)(2)!=INFINITY){

            //first transfer points in frame2 into frame1 coordinate
                    tmp=r0*range2(i,j)+t0;
                    tmpR=r0*normal2(i,j)+t0;

            //find the transfered point index in the frame1, note that it is a perspective project instead of ortho proj
                    u=tmp(1)/tmp(2)*(float)focalLen;
                    v=tmp(0)/tmp(2)*(float)focalLen;

                    u=m/2-u;
                    v+=n/2;



            //if the two points are 'similar' to each other, then take them into account
                    if(u>=0 && u<m && v>=0 && v<n){
                        regionCnt++;
                //cout<<"points fall into the valid region"<<endl;
                        if(range1(u,v)(2)!=INFINITY && normal1(u,v)(2)!=INFINITY){
                            if((range1(u,v)-tmp).dot(range1(u,v)-tmp)<ed){
                                disValCnt++;
                                if(normal1(u,v).dot(tmpR)/(normal1(u,v).squaredNorm()*tmpR.squaredNorm())>1-e0){
                                    norValCnt++;
                                    /*          */     G(0,1)=-tmp(2);      G(0,2)=tmp(1);
                                    G(1,0)=tmp(2);     /*          */       G(1,2)=-tmp(0);
                                    G(2,0)=-tmp(1);     G(2,1)=tmp(0);      /*          */
                                    cnt++;
                                    AT=G.transpose()*normal1(u,v);
                                    b=normal1(u,v).transpose()*(range1(u,v)-tmp);
                                    sumA+=AT*AT.transpose();
                                    sumATb+=AT*b;
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    else{		//if siftMerge==1 use sift match for frame merge
        int i1,i2,j1,j2;
        for(int i=0;i<good_matches.size();i++){
            i1=keypoint1[ good_matches[i].queryIdx].pt.y;
            j1=keypoint1[ good_matches[i].queryIdx].pt.x;
            i2=keypoint2[ good_matches[i].trainIdx].pt.y;
            j2=keypoint2[ good_matches[i].trainIdx].pt.x;



            if(range2(i2,j2)(2)!=INFINITY && normal2(i2,j2)(2)!=INFINITY){
                if(range1(i1,i1)(2)!=INFINITY && normal1(i1,i1)(2)!=INFINITY){
                    double factor=lamda*exp(pow(min_dist/good_matches[i].distance,2));
                    tmp=r0*range2(i2,j2)+t0;
                    tmpR=r0*normal2(i2,j2)+t0;
                    cnt++;
                    /*          */     G(0,1)=-tmp(2);      G(0,2)=tmp(1);
                    G(1,0)=tmp(2);     /*          */       G(1,2)=-tmp(0);
                    G(2,0)=-tmp(1);     G(2,1)=tmp(0);      /*          */
                    AT=G.transpose()*normal1(i1,j1);
                    b=normal1(i1,j1).transpose()*(range1(i1,j1)-tmp);
                    sumA+=factor*AT*AT.transpose();
                    sumATb+=factor*AT*b;
                }
            }
        }

    }

    cout<<"cnt:"<<cnt<<endl<<"regionCnt:"<<regionCnt<<endl<<"disValCnt:"<<disValCnt<<endl<<"norValCnt:"<<norValCnt<<endl;
    x=sumA.lu().solve(sumATb);
    /*          */          res(0,1)=x(2);          res(0,2)=-x(1);
    res(1,0)=-x(2);         /*          */          res(1,2)=x(0);
    res(2,0)=x(1);          res(2,1)=-x(0);         /*          */

    res(0,3)=x(3);
    res(1,3)=x(4);
    res(2,3)=x(5);
    return res;
}


//---------------------------------------------------------------------------
/////////////////////////  Idle Func  ///////////////////////////////////////
//---------------------------------------------------------------------------

void glutIdle (void)
{
    // Display the frame
    glutPostRedisplay();
}

//---------------------------------------------------------------------------
/////////////////////////  orientMe Func  ///////////////////////////////////
//---------------------------------------------------------------------------


void orientMe() {
    glLoadIdentity();
    //gluLookAt(foucsx-6*sin(angy),foucsy+6*sin(angx),foucsz+6-(6*(1-cos(angx))+6*(1-cos(angy))),foucsx,foucsy,foucsz,0.0f,1.0f,0.0f);
    gluLookAt(atx+view_radius*sin(angx)*cos(angy),aty+view_radius*sin(angx)*sin(angy),atz+view_radius*cos(angx),atx,aty,atz,0.0f,1.0f,0.0f);
}

void inputKey(int key, int x, int y) {
    switch (key) {
    case GLUT_KEY_LEFT :
        angy += 0.01f;
        orientMe();break;
    case GLUT_KEY_RIGHT :
        angy -=0.01f;
        orientMe();break;
    case GLUT_KEY_UP :
        angx += 0.01f;
        orientMe();break;
    case GLUT_KEY_DOWN :
        angx -= 0.01f;
        orientMe();break;
    case GLUT_KEY_PAGE_UP:
        angx = 0.0f;
        angx = 0.0f;
        angleX = 0.0;
        angleY = 0.0;
        orientMe();break;
    }
}

//---------------------------------------------------------------------------
/////////////////////////  KeyBoard Func  ///////////////////////////////////
//---------------------------------------------------------------------------

void glutKeyboard (unsigned char key, int x, int y)
{

    switch (key)
    {
    case 27:
        exit (1);
    case '1':
        g_nViewState = DISPLAY_MODE_OVERLAY;
        g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
        break;
    case '2':
        g_nViewState = DISPLAY_MODE_DEPTH;
        g_depth.GetAlternativeViewPointCap().ResetViewPoint();
        break;
    case '3':
        g_nViewState = DISPLAY_MODE_IMAGE;
        g_depth.GetAlternativeViewPointCap().ResetViewPoint();
        break;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // to be tested section begin
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    case 'u':       //save xyzdata to /media/E/future/ICP/globalRange1x.csv | globalRange1y.csv | globalRange1z.csv
        //save xyzdata to globalRange1x.csv
        fout.open("/media/E/future/ICP/globalRange1x.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata[i][j][0]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        //save xyzdata to globalRange1y.csv
        fout.open("/media/E/future/ICP/globalRange1y.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata[i][j][1]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        //save xyzdata to globalRange1z.csv
        fout.open("/media/E/future/ICP/globalRange1z.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata[i][j][2]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        break;
    case 'p':       //save xyzdata to /media/E/future/ICP/globalRange2x.csv | globalRange2y.csv | globalRange2z.csv
        //save xyzdata to globalRange2x.csv
        fout.open("/media/E/future/ICP/globalRange2x.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata2[i][j][0]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        //save xyzdata to globalRange1y.csv
        fout.open("/media/E/future/ICP/globalRange2y.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata2[i][j][1]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        //save xyzdata to globalRange1z.csv
        fout.open("/media/E/future/ICP/globalRange2z.csv");
        if(fout){
            for(int i = 0;i<height;i++){
                for(int j=0;j<width;j++)
                    fout<<xyzdata2[i][j][2]<<',';
                fout<<"\n";
            }
        }
        else{
            printf("canot open file.\n");
        }
        fout.close();
        break;
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // to be tested section end
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    case 'm':
        g_context.SetGlobalMirror(!g_context.GetGlobalMirror());
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //below case '5' is related with sift
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    case '5':
        // imshow("frame1",frame1);
        // imshow("frame2",frame2);
        // waitKey();
        detector.detect(frame1,keypoint1);
        detector.detect(frame2,keypoint2);
        extractor.compute(frame1,keypoint1,descriptor1);
        extractor.compute(frame2,keypoint2,descriptor2);
        matcher.match(descriptor1,descriptor2,matches);
        //calculate the min and max distance
        for( int i = 0; i < descriptor1.rows; i++ )
        { double dist = matches[i].distance;
          if( dist < min_dist ) min_dist = dist;
          if( dist > max_dist ) max_dist = dist;
        }

        cout<<"Maxdist:"<<max_dist<<endl;
        cout<<"Mindist:"<<min_dist<<endl;
        good_matches.clear();

        for( int i = 0; i < descriptor1.rows; i++ )
        { if( matches[i].distance <= max(2*min_dist, 0.02) )
          { good_matches.push_back( matches[i]); }
        }
        sort(good_matches.begin(), good_matches.end(), distance_comparator);

        drawMatches(frame1,keypoint1,frame2,keypoint2,good_matches,img_matches);
        imshow("matches",img_matches);
        //show info for good matches

//        test.x=10;
//        test.y=30;
//        circle( frame1, test\
//                , 5, Scalar(255,0,0),5);
//        circle( frame2,test\
//                , 5, Scalar(255,0,0),5);
          for( int i = 0; i < (int)good_matches.size(); i++ )
          {
              cout<<"====================================="<<endl;
              cout<<"-- Good Match ["<<i<<"] Keypoint 1: "<<good_matches[i].queryIdx<<"  -- Keypoint 2: "<<\
                    good_matches[i].trainIdx<<" -- Dist:"<<good_matches[i].distance<<endl;
              cout<<"img1:"<< keypoint1[ good_matches[i].queryIdx].pt// Query is first.
                    <<" matches "
                   <<"img2:"<< keypoint2[ good_matches[i].trainIdx].pt // Training is second.
                    <<endl;
//              circle( frame1, keypoint1[ good_matches[i].queryIdx].pt\
//                      , 5, Scalar(0,0,0),5);
//              circle( frame2, keypoint2[ good_matches[i].trainIdx].pt\
//                      , 5, Scalar(0,0,0),5);

          }
//          imshow("frame1",frame1);
//          imshow("frame2",frame2);


        waitKey();
        // destroyWindow("frame1");
        // destroyWindow("frame2");
        destroyWindow("matches");
        break;
    case'4' ://use sift match to merge two frames
        siftMerge=!siftMerge;
        if(siftMerge==1){
            cout<<"use SIFT match for frame merge"<<endl;
        }
        else{
            cout<<"use ICP for frame merge"<<endl;
        }
        break;
    case '7':   //toggle between use high / low definition for icp
        lowDefi=!lowDefi;
        if(lowDefi==1){
            focalLen=focalLen/2;
            cout<<"use low defi for icp"<<endl;
        }
        else{
            focalLen*=2;
            cout<<"use high defi for icp"<<endl;
        }
        break;
    case '6':
        cout<<"reset the rotation and translation mat"<<endl;
        r0=MatrixXf::Identity(3,3);
        t0=MatrixXf::Zero(3,1);
        break;
    case '9':   // toggle between use/ not use icp adjust
        if(icpAdjust==0){       //if previously not use icp, use icp right now

            std::cout<<"begin icp calculate"<<std::endl;
            std::cout<<"r0:"<<r0<<std::endl;
            std::cout<<"t0:"<<t0<<std::endl;
            std::cout<<"==================================="<<std::endl;

            if(lowDefi==0){
                getNormal(range1,normal1);
                getNormal(range2,normal2);
                rt=myICP(range1,normal1,range2,normal2,r0,t0,0.005,1);
            }
            else{
                getNormal(range1Low,normal1Low);
                getNormal(range2Low,normal2Low);
                //augFac*=1.4*regionCnt/(range2Low.rows()*range2Low.cols());
                augFac=1.1*regionCnt/(range2Low.rows()*range2Low.cols())*sqrt(augFac);
                float ed=edupper*(1-1.0*augFac*disValCnt/(regionCnt))+edlower*1.0*augFac*disValCnt/(regionCnt);
                if(ed>edupper)ed=edupper;
                if(ed<edlower)ed=edlower;
                float e0=e0upper*(1-1.0*augFac*norValCnt/disValCnt)+e0lower*1.0*augFac*norValCnt/disValCnt;
                if(e0>e0upper)e0=e0upper;
                if(e0<e0lower)e0=e0lower;
                rt=myICP(range1Low,normal1Low,range2Low,normal2Low,r0,t0,ed,e0);
                //regionCnt=8,disValCnt=4,norValCnt=2;

            }
            r0=rt.block<3,3>(0,0)*r0;
            t0+=rt.col(3);
            std::cout<<"end icp calculate"<<std::endl;
            std::cout<<"r0:"<<r0<<std::endl;
            std::cout<<"t0:"<<t0<<std::endl;
            std::cout<<"|||||||||||||||||||||||||||||||||||||||"<<std::endl;
            //store the merged range and normal frame in range2Merged and normal2Merged
            for(int i=0;i<height;i++){
                for(int j=0;j<width;j++){
                    if(range2(i,j)(2)!=INFINITY)
                        range2Merged(i,j)=r0*range2(i,j)+t0;
                    if(normal2(i,j)(2)!=INFINITY){
                        normal2Merged(i,j)=r0*normal2(i,j)+t0;
                    }
                }
            }
            icpAdjust=1;
        }
        else{       //if previously used icp, shut it down now
            icpAdjust=0;
        }
        break;
    case '0':   //iterate icp procedure
        std::cout<<"pressed 0"<<std::endl;
        std::cout<<"r0:"<<r0<<std::endl;
        std::cout<<"t0:"<<t0<<std::endl;
        std::cout<<"======================================"<<std::endl;
        if(lowDefi==0){
            rt=myICP(range1,normal1,range2,normal2,r0,t0,0.005,1);
        }
        else{
            //rt=myICP(range1Low,normal1Low,range2Low,normal2Low,r0,t0,0.03,1.2);
            augFac=1.1*regionCnt/(range2Low.rows()*range2Low.cols())*sqrt(augFac);
            float ed=edupper*(1-1.0*augFac*disValCnt/(regionCnt))+edlower*1.0*augFac*disValCnt/(regionCnt);
            if(ed>edupper)ed=edupper;
            if(ed<edlower)ed=edlower;
            float e0=e0upper*(1-1.0*augFac*norValCnt/disValCnt)+e0lower*1.0*augFac*norValCnt/disValCnt;
            if(e0>e0upper)e0=e0upper;
            if(e0<e0lower)e0=e0lower;
            rt=myICP(range1Low,normal1Low,range2Low,normal2Low,r0,t0,ed,e0);
        }
        r0=rt.block<3,3>(0,0)*r0;
        t0+=rt.col(3);
        std::cout<<"r0:"<<r0<<std::endl;
        std::cout<<"t0:"<<t0<<std::endl;
        std::cout<<"|||||||||||||||||||||||||||||||||||||||"<<std::endl;
        for(int i=0;i<height;i++){
            for(int j=0;j<width;j++){
                if(range2(i,j)(2)!=INFINITY)
                    range2Merged(i,j)=r0*range2(i,j)+t0;
                if(normal2(i,j)(2)!=INFINITY){
                    normal2Merged(i,j)=r0*normal2(i,j)+t0;
                }
            }
        }
        break;
    case '8':
        primeFlag=!primeFlag;
        break;
    case 'y':       //save current range frame to RawRange1.csv
        fout.open("/media/E/future/ICP/RawRange1.csv");
        if(fout){
            const XnDepthPixel* pDepth = g_depthMD.Data();
            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
                {
                    fout<<*pDepth<<',';
                }
                fout<<"\n";
            }
        }
        else
            printf("canot open file.\n");
        //qDebug()<<"canot open file.\n";

        fout.close();
        break;

    case 't':       //save current range frame to RawRange2.csv
        fout.open("/media/E/future/ICP/RawRange2.csv");
        if(fout){
            const XnDepthPixel* pDepth = g_depthMD.Data();
            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
                {
                    fout<<*pDepth<<',';
                }
                fout<<"\n";
            }
        }
        else
            printf("canot open file.\n");
        //qDebug()<<"canot open file.\n";
        fout.close();
        break;

    case 'q':       //convert current range fame to global coordinate and change dispaly mode
        g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);

        // Read a new frame
        rc = g_context.WaitAnyUpdateAll();
        if (rc != XN_STATUS_OK)
        {
            printf("Read failed: %s\n", xnGetStatusString(rc));
            return;
        }

        g_depth.GetMetaData(g_depthMD);
        g_image.GetMetaData(g_imageMD);
        std::cout<<"toggle global"<<std::endl;
        mindepth=INFINITY;
        maxdepth=-1;
        xnOSMemSet(g_pDepthHist, 0, MAX_DEPTH*sizeof(float));
        //unsigned int nNumberOfPoints = 0;
        if(displayMode==0){
            eyex=0;
            eyey=0;
            eyez=0;
            const XnDepthPixel* pDepth = g_depthMD.Data();
            sumx=0;
            sumy=0;
            sumz=0;
            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
                {
                    /////////////////////////////////////////////////////////////////////////////////////////////////
                    //##############    continuted to be done: 1. check the validirity 2 apply texture array
                    /////////////////////////////////////////////////////////////////////////////////////////////////
                    if (*pDepth == 0)
                    {
                        xyzdata[y][x][0]=INFINITY;
                        xyzdata[y][x][1]=INFINITY;
                        xyzdata[y][x][2]=INFINITY;
                        range1(y,x)(0)=INFINITY;
                        range1(y,x)(1)=INFINITY;
                        range1(y,x)(2)=INFINITY;
                        continue;
                    }
                    XnPoint3D proj, real;
                    proj.X = x;
                    proj.Y = y;
                    proj.Z = *pDepth;
                    g_pDepthHist[*pDepth]++;
                    nNumberOfPoints++;
                    g_depth.ConvertProjectiveToRealWorld(1, &proj, &real);
                    // from mm to meters
                    //pointCloud_XYZ.at(y,x) = cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f);
                    xyzdata[y][x][0]=real.X*0.001f;
                    xyzdata[y][x][1]=real.Y*0.001f;
                    xyzdata[y][x][2]=real.Z*0.001f;
                    range1(y,x)(0)=xyzdata[y][x][0];
                    range1(y,x)(1)=xyzdata[y][x][1];
                    range1(y,x)(2)=xyzdata[y][x][2];

                    sumx+=xyzdata[y][x][0];
                    sumy+=xyzdata[y][x][1];
                    sumz+=xyzdata[y][x][2];


                    if(xyzdata[y][x][2]>maxdepth)
                        maxdepth=xyzdata[x][y][2];
                    if(xyzdata[y][x][2]!=0&&xyzdata[y][x][2]<mindepth)
                        mindepth=xyzdata[x][y][2];

                }
            }
            Vector3f tmp;
            int cnt=0;
            for(int i=0;i<range1Low.rows();i++){
                for(int j=0;j<range1Low.cols();j++){
                    cnt=0;
                    tmp=MatrixXf::Zero(3,1);
                    if(range1(2*i,2*j)(2)!=INFINITY){
                        cnt++;
                        tmp+=range1(2*i,2*j);
                    }
                    if(range1(2*i+1,2*j)(2)!=INFINITY){
                        cnt++;
                        tmp+=range1(2*i+1,2*j);
                    }
                    if(range1(2*i,2*j+1)(2)!=INFINITY){
                        cnt++;
                        tmp+=range1(2*i,2*j+1);
                    }
                    if(range1(2*i+1,2*j+1)(2)!=INFINITY){
                        cnt++;
                        tmp+=range1(2*i+1,2*j+1);
                    }
                    if(cnt<=1){
                        range1Low(i,j)(0)=INFINITY;
                        range1Low(i,j)(1)=INFINITY;
                        range1Low(i,j)(2)=INFINITY;
                    }
                    else
                        range1Low(i,j)=tmp/cnt;
                }
            }
            avgx=sumx/nNumberOfPoints;
            avgy=sumy/nNumberOfPoints;
            avgz=sumz/nNumberOfPoints;
            atx=avgx;
            aty=avgy;
            atz=avgz;

            //std::cout<<"maxdepth:"<<maxdepth<<std::endl;
            //std::cout<<"mindepth:"<<mindepth<<std::endl;

            const XnRGB24Pixel* pImage = g_imageMD.RGB24Data();
            for (XnUInt y = 0; y < g_imageMD.YRes(); ++y)
            {

                for (XnUInt x = 0; x < g_imageMD.XRes(); ++x,++pImage)
                {
                    //pointCloud_XYZ.at(y,x) = cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f);
                    texture[y][x][0]=pImage->nRed;
                    texture[y][x][1]=pImage->nGreen;
                    texture[y][x][2]=pImage->nBlue;
                    //frame1.at<Vec3b>(y,x)=pImage->nBlue<<16+pImage->nGreen<<8+pImage->nRed;
                    frame1.at<Vec3b>(y,x)[0]=pImage->nBlue;
                    frame1.at<Vec3b>(y,x)[1]=pImage->nGreen;
                    frame1.at<Vec3b>(y,x)[2]=pImage->nRed;

                }

            }

            displayMode=1;

        }
        else
            displayMode=0;

        break;
    case 'r' :      //merge the two frame
        g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
        if(mergeFlag==0&&displayMode==0){
//            tranMat[0]=0;
//            tranMat[1]=0;
//            tranMat[2]=2;
            rotMat[0][0]=cos(beta);     rotMat[0][1]=0;         rotMat[0][2]=-sin(beta);
            rotMat[1][0]=0;             rotMat[1][1]=1;         rotMat[1][2]=0;
            rotMat[2][0]=sin(beta);     rotMat[2][1]=0;         rotMat[2][2]=cos(beta);

            XnStatus rc = XN_STATUS_OK;

            // Read a new frame
            rc = g_context.WaitAnyUpdateAll();
            if (rc != XN_STATUS_OK)
            {
                printf("Read failed: %s\n", xnGetStatusString(rc));
                return;
            }

            g_depth.GetMetaData(g_depthMD);
            g_image.GetMetaData(g_imageMD);

            const XnDepthPixel* pDepth = g_depthMD.Data();
            sumx=0;
            sumy=0;
            sumz=0;
            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
                {
                    /////////////////////////////////////////////////////////////////////////////////////////////////
                    //##############    continuted to be done: 1. check the validirity 2 apply texture array
                    /////////////////////////////////////////////////////////////////////////////////////////////////
                    if (*pDepth == 0)
                    {
                        xyzdata2[y][x][0]=INFINITY;
                        xyzdata2[y][x][1]=INFINITY;
                        xyzdata2[y][x][2]=INFINITY;
                        range2(y,x)(0)=INFINITY;
                        range2(y,x)(1)=INFINITY;
                        range2(y,x)(2)=INFINITY;
                        continue;
                    }
                    XnPoint3D proj, real;
                    proj.X = x;
                    proj.Y = y;
                    proj.Z = *pDepth;
                    g_pDepthHist[*pDepth]++;
                    nNumberOfPoints++;
                    g_depth.ConvertProjectiveToRealWorld(1, &proj, &real);
                    // from mm to meters
                    //pointCloud_XYZ.at(y,x) = cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f);
                    real.X*=0.001f;
                    real.Y*=0.001f;
                    real.Z*=0.001f;
                    xyzdata2[y][x][0]=real.X*rotMat[0][0]+real.Y*rotMat[0][1]+real.Z*rotMat[0][2]+tranMat[0];
                    xyzdata2[y][x][1]=real.X*rotMat[1][0]+real.Y*rotMat[1][1]+real.Z*rotMat[1][2]+tranMat[1];
                    xyzdata2[y][x][2]=real.X*rotMat[2][0]+real.Y*rotMat[2][1]+real.Z*rotMat[2][2]+tranMat[2];
                    range2(y,x)(0)=xyzdata2[y][x][0];
                    range2(y,x)(1)=xyzdata2[y][x][1];
                    range2(y,x)(2)=xyzdata2[y][x][2];

                    sumx+=xyzdata2[y][x][0];
                    sumy+=xyzdata2[y][x][1];
                    sumz+=xyzdata2[y][x][2];


                    if(xyzdata2[y][x][2]>maxdepth)
                        maxdepth=xyzdata2[x][y][2];
                    if(xyzdata2[y][x][2]!=0&&xyzdata2[y][x][2]<mindepth)
                        mindepth=xyzdata2[x][y][2];

                }
            }

            Vector3f tmp;
            int cnt=0;
            for(int i=0;i<range2Low.rows();i++){
                for(int j=0;j<range2Low.cols();j++){
                    cnt=0;
                    tmp=MatrixXf::Zero(3,1);
                    if(range2(2*i,2*j)(2)!=INFINITY){
                        cnt++;
                        tmp+=range2(2*i,2*j);
                    }
                    if(range2(2*i+1,2*j)(2)!=INFINITY){
                        cnt++;
                        tmp+=range2(2*i+1,2*j);
                    }
                    if(range2(2*i,2*j+1)(2)!=INFINITY){
                        cnt++;
                        tmp+=range2(2*i,2*j+1);
                    }
                    if(range2(2*i+1,2*j+1)(2)!=INFINITY){
                        cnt++;
                        tmp+=range2(2*i+1,2*j+1);
                    }
                    if(cnt<=1){
                        range2Low(i,j)(0)=INFINITY;
                        range2Low(i,j)(1)=INFINITY;
                        range2Low(i,j)(2)=INFINITY;
                    }
                    else
                    range2Low(i,j)=tmp/cnt;
                }
            }
            avgx=sumx/nNumberOfPoints;
            avgy=sumy/nNumberOfPoints;
            avgz=sumz/nNumberOfPoints;

            //std::cout<<"maxdepth:"<<maxdepth<<std::endl;
            //std::cout<<"mindepth:"<<mindepth<<std::endl;

            const XnRGB24Pixel* pImage = g_imageMD.RGB24Data();
            for (XnUInt y = 0; y < g_imageMD.YRes(); ++y)
            {

                for (XnUInt x = 0; x < g_imageMD.XRes(); ++x,++pImage)
                {
                    //pointCloud_XYZ.at(y,x) = cv::Point3f( real.X*0.001f, real.Y*0.001f, real.Z*0.001f);
                    texture2[y][x][0]=pImage->nRed;
                    texture2[y][x][1]=pImage->nGreen;
                    texture2[y][x][2]=pImage->nBlue;
                    frame2.at<Vec3b>(y,x)[0]=pImage->nBlue;
                    frame2.at<Vec3b>(y,x)[1]=pImage->nGreen;
                    frame2.at<Vec3b>(y,x)[2]=pImage->nRed;
                }

            }


            //mergeFlag=1;
        }
        else
            mergeFlag=!mergeFlag;
        //std::cout<<"rotMat[0][0]="<<rotMat[0][0]<<std::endl;
        //std::cout<<"beta="<<beta<<std::endl;
        std::cout<<"displayMode:"<<displayMode<<endl<<"icpAdjust:"<<icpAdjust<<endl<<"mergeFlag:"<<mergeFlag<<std::endl;

        break;

    case 'a' :
        eyex-=0.2;
        //orientMe();
        break;
    case 'd' :
        eyex+=0.2;
        //orientMe();
        break;
    case 'w' :
        eyey+=0.2;
        //orientMe();
        break;

    case 's' :
        eyey-=0.2;
        //orientMe();
        break;
    case 'z' :
        eyez+=0.2;
        //orientMe();
        break;
    case 'x' :
        eyez-=0.2;
        //orientMe();
        break;
    case 'e':
        eyex=avgx;
        eyey=avgy;
        eyez=avgz;
        break;

    /////////// control 'center' point
    case 'j' :
        atx-=0.2;
        //orientMe();
        break;
    case 'l' :
        atx+=0.2;
        //orientMe();
        break;
    case 'i' :
        aty+=0.2;
        //orientMe();
        break;

    case 'k' :
        aty-=0.2;
        //orientMe();
        break;
    case 'b' :
        atz+=0.2;
        //orientMe();
        break;
    case 'n' :
        atz-=0.2;
        //orientMe();
        break;
    case 'o':
        atx=0;
        aty=0;
        atz=0;
        break;
    }
}

//---------------------------------------------------------------------------
/////////////////////////  Mouse click Func  ///////////////////////////////////
//---------------------------------------------------------------------------

void mouse(int button, int state, int x, int y)
{
    if(button == GLUT_LEFT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            leftClickHold=true;
        }
        else
        {
            leftClickHold=false;
        }
    }
    if (button== GLUT_RIGHT_BUTTON)
    {
        if(state == GLUT_DOWN)
        {
            rightClickHold=true;
        }
        else
        {
            rightClickHold=false;
        }
    }
}
//---------------------------------------------------------------------------
/////////////////////////  Mouse motion Func  ///////////////////////////////////
//---------------------------------------------------------------------------

void motion(int x, int y)
{
    int rstep = 5;
    if(leftClickHold==true)
    {
        if( abs(x-mx) > abs(y-my) )
        {
            rx += SIGN(x-mx)*rstep;
        }
        else
        {
            ry -= SIGN(y-my)*rstep;
        }

        mx=x;
        my=y;
        glutPostRedisplay();
    }
    if(rightClickHold==true)
    {
        if( y-my > 0 )
        {
            radius += 100.0;
        }
        else if( y-my < 0 )
        {
            radius -= 100.0;
        }
        radius = std::max( radius, 100.0 );
        mx=x;
        my=y;
        glutPostRedisplay();
    }
}

//---------------------------------------------------------------------------
/////////////////////////  calSimilar Func  /////////////////////////////////
//---------------------------------------------------------------------------

bool calSimilar(float *x, float *y){
    float sum=0;
    for(int i=0;i<3;i++)
        sum+=pow(x[i]-y[i],2);
    if(sum<thres)
        return true;
    return false;

}
//---------------------------------------------------------------------------
/////////////////////////  Display Func  ///////////////////////////////////
//---------------------------------------------------------------------------

void glutDisplay (void)
{
    // if displaymode==0 then display range image
    if(displayMode==0){

        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        // Reset the coordinate system before modifying
        glLoadIdentity();
        // set the camera position


            XnStatus rc = XN_STATUS_OK;

            // Read a new frame
            rc = g_context.WaitAnyUpdateAll();
            if (rc != XN_STATUS_OK)
            {
                printf("Read failed: %s\n", xnGetStatusString(rc));
                return;
            }

            g_depth.GetMetaData(g_depthMD);
            g_image.GetMetaData(g_imageMD);

            const XnDepthPixel* pDepth = g_depthMD.Data();
            const XnUInt8* pImage = g_imageMD.Data();

            unsigned int nImageScale = GL_WIN_SIZE_X / g_depthMD.FullXRes();

            // Copied from SimpleViewer
            // Clear the OpenGL buffers
            glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Setup the OpenGL viewpoint
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0, GL_WIN_SIZE_X, GL_WIN_SIZE_Y, 0, -1.0, 1.0);

            // Calculate the accumulative histogram (the yellow display...)
            xnOSMemSet(g_pDepthHist, 0, MAX_DEPTH*sizeof(float));

            unsigned int nNumberOfPoints = 0;
            for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
            {
                for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth)
                {
                    if (*pDepth != 0)
                    {
                        g_pDepthHist[*pDepth]++;
                        nNumberOfPoints++;
                    }
                }
            }
            for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
            {
                g_pDepthHist[nIndex] += g_pDepthHist[nIndex-1];
            }
            if (nNumberOfPoints)
            {
                for (int nIndex=1; nIndex<MAX_DEPTH; nIndex++)
                {
                    g_pDepthHist[nIndex] = (unsigned int)(256 * (1.0f - (g_pDepthHist[nIndex] / nNumberOfPoints)));
                }
            }

            xnOSMemSet(g_pTexMap, 0, g_nTexMapX*g_nTexMapY*sizeof(XnRGB24Pixel));

            // check if we need to draw image frame to texture
            if (g_nViewState == DISPLAY_MODE_OVERLAY ||
                g_nViewState == DISPLAY_MODE_IMAGE)
            {
                const XnRGB24Pixel* pImageRow = g_imageMD.RGB24Data();
                XnRGB24Pixel* pTexRow = g_pTexMap + g_imageMD.YOffset() * g_nTexMapX;

                for (XnUInt y = 0; y < g_imageMD.YRes(); ++y)
                {
                    const XnRGB24Pixel* pImage = pImageRow;
                    XnRGB24Pixel* pTex = pTexRow + g_imageMD.XOffset();

                    for (XnUInt x = 0; x < g_imageMD.XRes(); ++x, ++pImage, ++pTex)
                    {
                        *pTex = *pImage;
                    }

                    pImageRow += g_imageMD.XRes();
                    pTexRow += g_nTexMapX;
                }
            }

            // check if we need to draw depth frame to texture
            if (g_nViewState == DISPLAY_MODE_OVERLAY ||
                g_nViewState == DISPLAY_MODE_DEPTH)
            {
                const XnDepthPixel* pDepthRow = g_depthMD.Data();
                XnRGB24Pixel* pTexRow = g_pTexMap + g_depthMD.YOffset() * g_nTexMapX;

                for (XnUInt y = 0; y < g_depthMD.YRes(); ++y)
                {
                    const XnDepthPixel* pDepth = pDepthRow;
                    XnRGB24Pixel* pTex = pTexRow + g_depthMD.XOffset();

                    for (XnUInt x = 0; x < g_depthMD.XRes(); ++x, ++pDepth, ++pTex)
                    {
                        if (*pDepth != 0)
                        {
                            int nHistValue = g_pDepthHist[*pDepth];
                            pTex->nRed = nHistValue;
                            pTex->nGreen = nHistValue;
                            pTex->nBlue = 0;
                        }
                    }

                    pDepthRow += g_depthMD.XRes();
                    pTexRow += g_nTexMapX;
                }
            }

            // Create the OpenGL texture map
            glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, g_nTexMapX, g_nTexMapY, 0, GL_RGB, GL_UNSIGNED_BYTE, g_pTexMap);

            // Display the OpenGL texture map
            glColor4f(1,1,1,1);

            glBegin(GL_QUADS);

            int nXRes = g_depthMD.FullXRes();
            int nYRes = g_depthMD.FullYRes();

            // upper left
            glTexCoord2f(0, 0);
            glVertex2f(0, 0);
            // upper right
            glTexCoord2f((float)nXRes/(float)g_nTexMapX, 0);
            glVertex2f(GL_WIN_SIZE_X, 0);
            // bottom right
            glTexCoord2f((float)nXRes/(float)g_nTexMapX, (float)nYRes/(float)g_nTexMapY);
            glVertex2f(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);
            // bottom left
            glTexCoord2f(0, (float)nYRes/(float)g_nTexMapY);
            glVertex2f(0, GL_WIN_SIZE_Y);

            glEnd();
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //##############  begin plotting triangles     ########    continuted to be done
    /////////////////////////////////////////////////////////////////////////////////////////////////

    //if displaymode==1 display global model
    else{

        glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        // Reset the coordinate system before modifying
        //glLoadIdentity();
        // set the camera position


//        atx = 0.0f;
//        aty = 0.0f;
//        atz = ( mindepth - maxdepth ) / 2.0f;
//        eyex = atx + radius * sin( PI * ry / 180.0f ) * cos( PI * rx/ 180.0f );
//        eyey = aty + radius * cos( PI * ry/ 180.0f );
//        eyez = atz + radius * sin( PI * ry / 180.0f ) * sin( PI * rx/ 180.0f );
//        gluLookAt (eyex, eyey, eyez, atx, aty, atz, 0.0, 1.0, 0.0);
//        glRotatef(0,0,1,0);
//        glRotatef(-180,1,0,0);


        glLoadIdentity();
        gluPerspective(100,1,0.01,3000);
        gluLookAt (eyex, eyey, eyez, atx, aty, atz, 0.0, 1.0, 0.0);

        //begin plotting the triangels
        if(primeFlag==1)
        for (int i = 0; i < height; i++){
        {
                //glBegin(GL_TRIANGLE_STRIP);
                for (int j = 0; j < width; j++)
                {
                        // for each vertex, we calculate the vertex color,
                        // we set the texture coordinate, and we draw the vertex.
                        /*
                        the vertexes are drawn in this order:
                        0 ---> 1
                        /
                        /
                        /
                        2 ---> 3
                        */

                        // draw vertex 0
                    float cdepth=xyzdata[i][j][2];
                    float thres=100;

                    glBegin(GL_TRIANGLE_STRIP);

                    if(xyzdata[i][j][0]!=INFINITY){
                        glTexCoord2f(0.0f, 0.0f);
                        glColor3f(texture[i][j][0]/255.0f, texture[i][j][1]/255.0f, texture[i][j][2]/255.0f);
                        glVertex3f(xyzdata[i][j][0], xyzdata[i][j][1], xyzdata[i][j][2]);
                    }
                        // draw vertex 1
                    //if(xyzdata[i+1][j][0]!=INFINITY){
                    if(calSimilar(xyzdata[i][j],xyzdata[i+1][j])){
                        glTexCoord2f(1.0f, 0.0f);
                        glColor3f(texture[i+1][j][0]/255.0f, texture[i+1][j][1]/255.0f, texture[i+1][j][2]/255.0f);
                        glVertex3f(xyzdata[i+1][j][0], xyzdata[i+1][j][1], xyzdata[i+1][j][2]);
                    }
                        // draw vertex 2
                    //if(xyzdata[i][j+1][0]!=INFINITY){
                    if(calSimilar(xyzdata[i][j],xyzdata[i][j+1])){
                        glTexCoord2f(0.0f, 1.0f);
                        glColor3f(texture[i][j+1][0]/255.0f, texture[i][j+1][1]/255.0f, texture[i][j+1][2]/255.0f);
                        glVertex3f(xyzdata[i][j+1][0], xyzdata[i][j+1][1], xyzdata[i][j+1][2]);
                    }
                        // draw vertex 3
                    //if(xyzdata[i+1][j+1][0]!=INFINITY){
                    if(calSimilar(xyzdata[i][j],xyzdata[i+1][j+1])){
                        glTexCoord2f(1.0f, 1.0f);
                        glColor3f(texture[i+1][j+1][0]/255.0f, texture[i+1][j+1][1]/255.0f, texture[i+1][j+1][2]/255.0f);
                        glVertex3f(xyzdata[i+1][j+1][0], xyzdata[i+1][j+1][1], xyzdata[i+1][j+1][2]);
                    }
                    glEnd();
                }
                //glEnd();

         }
    }

        if(mergeFlag==1){
            //if icpAdjst is off, merge the two image
            if(icpAdjust==0){
            //std::cout<<"enter the merge stage"<<std::endl;
                for (int i = 0; i < height; i++)
                {
                        //glBegin(GL_TRIANGLE_STRIP);
                        for (int j = 0; j < width; j++)
                        {
                                // for each vertex, we calculate the vertex color,
                                // we set the texture coordinate, and we draw the vertex.
                                /*
                                the vertexes are drawn in this order:
                                0 ---> 1
                                /
                                /
                                /
                                2 ---> 3
                                */

                                // draw vertex 0
                            glBegin(GL_TRIANGLE_STRIP);
                            float cdepth=xyzdata2[i][j][2];
                            float thres=100;
                            if(xyzdata2[i][j][0]!=INFINITY){
                                glTexCoord2f(0.0f, 0.0f);
                                glColor3f(texture2[i][j][0]/255.0f, texture2[i][j][1]/255.0f, texture2[i][j][2]/255.0f);
                                glVertex3f(xyzdata2[i][j][0], xyzdata2[i][j][1], xyzdata2[i][j][2]);
                            }
                                // draw vertex 1
                            //if(xyzdata[i+1][j][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i+1][j])){
                                glTexCoord2f(1.0f, 0.0f);
                                glColor3f(texture2[i+1][j][0]/255.0f, texture2[i+1][j][1]/255.0f, texture2[i+1][j][2]/255.0f);
                                glVertex3f(xyzdata2[i+1][j][0], xyzdata2[i+1][j][1], xyzdata2[i+1][j][2]);
                            }
                                // draw vertex 2
                            //if(xyzdata[i][j+1][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i][j+1])){
                                glTexCoord2f(0.0f, 1.0f);
                                glColor3f(texture2[i][j+1][0]/255.0f, texture2[i][j+1][1]/255.0f, texture2[i][j+1][2]/255.0f);
                                glVertex3f(xyzdata2[i][j+1][0], xyzdata2[i][j+1][1], xyzdata2[i][j+1][2]);
                            }
                                // draw vertex 3
                            //if(xyzdata[i+1][j+1][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i+1][j+1])){
                                glTexCoord2f(1.0f, 1.0f);
                                glColor3f(texture2[i+1][j+1][0]/255.0f, texture2[i+1][j+1][1]/255.0f, texture2[i+1][j+1][2]/255.0f);
                                glVertex3f(xyzdata2[i+1][j+1][0], xyzdata2[i+1][j+1][1], xyzdata2[i+1][j+1][2]);
                            }
                            glEnd();
                        }
                        //glEnd();

                 }
             }
             else{          //if icpAdjust is on, draw the icpAdjsuted clouds
                //cout<<"start to draw icp clouds"<<endl;
                for (int i = 0; i < height; i++)
                    {
                        //glBegin(GL_TRIANGLE_STRIP);
                        for (int j = 0; j < width; j++)
                        {
                                //Vertex3f tmp;
                                // draw vertex 0
                            glBegin(GL_TRIANGLE_STRIP);
                            float cdepth=xyzdata2[i][j][2];
                            float thres=100;
                            //tmp=r0*range2(i,j)+t0;
                            if(range2Merged(i,j)(2)!=INFINITY){
                                glTexCoord2f(0.0f, 0.0f);
                                glColor3f(texture2[i][j][0]/255.0f, texture2[i][j][1]/255.0f, texture2[i][j][2]/255.0f);
                                glVertex3f(range2Merged(i,j)(0), range2Merged(i,j)(1), range2Merged(i,j)(2));
                            }
                                // draw vertex 1
                            //if(xyzdata[i+1][j][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i+1][j])){
                                glTexCoord2f(1.0f, 0.0f);
                                glColor3f(texture2[i+1][j][0]/255.0f, texture2[i+1][j][1]/255.0f, texture2[i+1][j][2]/255.0f);
                                glVertex3f(range2Merged(i+1,j)(0), range2Merged(i+1,j)(1), range2Merged(i+1,j)(2));
                            }
                                // draw vertex 2
                            //if(xyzdata[i][j+1][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i][j+1])){
                                glTexCoord2f(0.0f, 1.0f);
                                glColor3f(texture2[i][j+1][0]/255.0f, texture2[i][j+1][1]/255.0f, texture2[i][j+1][2]/255.0f);
                                glVertex3f(range2Merged(i,j+1)(0), range2Merged(i,j+1)(1), range2Merged(i,j+1)(2));
                            }
                                // draw vertex 3
                            //if(xyzdata[i+1][j+1][0]!=INFINITY){
                            if(calSimilar(xyzdata2[i][j],xyzdata2[i+1][j+1])){
                                glTexCoord2f(1.0f, 1.0f);
                                glColor3f(texture2[i+1][j+1][0]/255.0f, texture2[i+1][j+1][1]/255.0f, texture2[i+1][j+1][2]/255.0f);
                                glVertex3f(range2Merged(i+1,j+1)(0), range2Merged(i+1,j+1)(1), range2Merged(i+1,j+1)(2));
                            }
                            glEnd();
                        }
                        //glEnd();

                    }
                //cout<<"end draw icp clouds"<<endl;
             }
        }
        //std::cout<<"leave the merge stage"<<std::endl;

//        glPushMatrix();
//        glTranslatef(atx,aty,atz);
//        glColor3f(1,0,0);
//        glutSolidSphere (0.5, 20, 30);
//        glPopMatrix();
        //cout<<"begin draw axis"<<endl;
        glColor3f(1,0,0);
        glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(0,50,0);
            glVertex3f(0,0,0);
            glVertex3f(50,0,0);
            glVertex3f(0,0,0);
            glVertex3f(0,0,50);
        glEnd();

        // enable blending
        glEnable(GL_BLEND);
        // enable read-only depth buffer
        glDepthMask(GL_FALSE);
        // set the blend function to what we use for transparency
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        // set back to normal depth buffer mode (writable)
        glDepthMask(GL_TRUE);
        // disable blending
        glDisable(GL_BLEND);
        /* float x,y,z;
        // 绘制图像点云
        glPointSize(1.0);
        glBegin(GL_POINTS);
        for (int i=0;i<height;i++){
        for (int j=0;j<width;j++){
        // color interpolation
        glColor3f(texture[i][j][0]/255, texture[i][j][1]/255, texture[i][j][2]/255);
        x= xyzdata[i][j][0];
        y= xyzdata[i][j][1];
        z= xyzdata[i][j][2];
        glVertex3f(x,y,z);
        }
        }
        glEnd(); */
        glFlush();
        glutSwapBuffers();

        //end plotting the triangels
    }
    /////////////////////////////////////////////////////////////////////////////////////////////////
    //##############  end plotting triangles     ########    continuted to be done
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Swap the OpenGL display buffers
    glutSwapBuffers();
}

//---------------------------------------------------------------------------
///////////////////////// NIInit Func  ///////////////////////////////////////
//---------------------------------------------------------------------------
int NIInit()
{
    EnumerationErrors errors;
    rc = g_context.InitFromXmlFile(SAMPLE_XML_PATH, g_scriptNode, &errors);
    if (rc == XN_STATUS_NO_NODE_PRESENT)
    {
        XnChar strError[1024];
        errors.ToString(strError, 1024);
        printf("%s\n", strError);
        return -1;
    }
    else if (rc != XN_STATUS_OK)
    {
        printf("Open failed: %s\n", xnGetStatusString(rc));
        return -1;
    }

    rc = g_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
    if (rc != XN_STATUS_OK)
    {
        printf("No depth node exists! Check your XML.");
        return -1;
    }

    rc = g_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
    if (rc != XN_STATUS_OK)
    {
        printf("No image node exists! Check your XML.");
        return -1;
    }
    return 1;
}

//---------------------------------------------------------------------------
/////////////////////////  checkNIData Func  ////////////////////////////////
//---------------------------------------------------------------------------
int checkNIData()
{
    // Hybrid mode isn't supported in this sample
    if (g_imageMD.FullXRes() != g_depthMD.FullXRes() || g_imageMD.FullYRes() != g_depthMD.FullYRes())
    {
        printf ("The device depth and image resolution must be equal!\n");
        return -1;
    }

    // RGB is the only image format supported.
    if (g_imageMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
    {
        printf("The device image format must be RGB24\n");
        return -1;
    }
    return 1;
}

//---------------------------------------------------------------------------
/////////////////////////  Main Func  ///////////////////////////////////////
//---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    //Initial OpenNI
    if(NIInit()<0)
        return rc;
    g_depth.GetAlternativeViewPointCap().SetViewPoint( g_image );

    g_depth.GetMetaData(g_depthMD);
    g_image.GetMetaData(g_imageMD);


    //g_depth.ConvertProjectiveToRealWorld()

    //Check OpenNI data format
    if(checkNIData()<0)
        return 1;


    // Texture map init
    g_nTexMapX = (((unsigned short)(g_depthMD.FullXRes()-1) / 512) + 1) * 512;
    g_nTexMapY = (((unsigned short)(g_depthMD.FullYRes()-1) / 512) + 1) * 512;
    g_pTexMap = (XnRGB24Pixel*)malloc(g_nTexMapX * g_nTexMapY * sizeof(XnRGB24Pixel));

    //manual info
    std::cout<<"Welcome\
        \nBelow is the manual of this applicaiton:\
        \n'u':save xyzdata to globalRange1x|y|z.csv\
        \n'p':save xyzdata to globalRange2x|y|z.csv\
        \n'm':change to the mirror view\
        \n'y':save current range frame to RawRange1.csv\
        \n't':save current range frame to RawRange2.csv\
        \n'q':convert current frame to global coordinate, change dispaly mode\
        \n'r':merge the two frame\
        \n'a':eyex--\
        \n'd':eyex++\
        \n'w':eyey++\
        \n's':eyey--\
        \n'z':eyez++\
        \n'x':eyez--\
        \n'e':set eye to original positon\
        \n'j':atx--\
        \n'l':atx++\
        \n'i':aty++\
        \n'k':aty--\
        \n'b':atz++\
        \n'n':atz--\
        \n'o':set atx|y|z to original\
        \n'9':toggle between use/not icp to adjust merge\
        \n'0':iterate icp procedure\
        \n'8':toggle between disp | not prime frame\
        \n'7':toggle between use high|low resolution for icp\
        \n'6':reset the rotation and translation mat\
        \n'5':show sift keypoits of two range frame\
        \n'4':use sift match to merge two frames\
        "<<std::endl;
    // OpenGL init
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    //glutInitWindowSize(GL_WIN_SIZE_X, GL_WIN_SIZE_Y);

    //glutFullScreen();
    glutInitWindowPosition(10,320);
    glutInitWindowSize(glWinWidth, glWinHeight);
    glutCreateWindow ("My Kinect Viewer");
    //glutSetCursor(GLUT_CURSOR_NONE);

    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutKeyboardFunc(glutKeyboard);
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutIdle);
    //glutSpecialFunc(inputKey);

    //glutPostRedisplay();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);

    // Per frame code is in glutDisplay
    glutMainLoop();

    return 0;
}

