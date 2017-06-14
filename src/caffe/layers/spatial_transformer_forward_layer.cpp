#include <cmath>
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/custom_layers.hpp"

namespace caffe {

    template <typename Dtype>
    void SpatialTransformerForwardLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {
    }

    template <typename Dtype>
    void SpatialTransformerForwardLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

        //bottom: "up_bbox_gt"
        //bottom: "down_bbox_gt"
        //bottom: "full_bbox_gt"
        //bottom: "theta"
        //top: "up_bbox_targets"
        //top: "down_bbox_targets"
        //top: "full_bbox_targets"

        top[0]->ReshapeLike(*bottom[0]);
        top[1]->ReshapeLike(*bottom[1]);
        top[2]->ReshapeLike(*bottom[2]);

        CHECK_EQ(bottom[3]->shape(1), 4) << "Second blob should be 4-dimension theta";
        
        vector<int> theta_shape(2);
        theta_shape[0] = 2;
        theta_shape[1] = 3;

        theta_.Reshape(theta_shape);

        vector<int> coordinate_shape(2);
        coordinate_shape[0] = 3;
        coordinate_shape[1] = 2;

        coordinate_.Reshape(coordinate_shape);        

    }

    template <typename Dtype>
    void SpatialTransformerForwardLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {

        int batch_size = bottom[1]->shape(0);
        Dtype* top_data_up = top[0]->mutable_cpu_data();
        Dtype* top_data_down = top[1]->mutable_cpu_data();
        Dtype* top_data_full = top[2]->mutable_cpu_data();
        Dtype* theta_blob_data = theta_.mutable_cpu_data();
        Dtype* coordinate_blob_data = coordinate_.mutable_cpu_data();
        const Dtype* bottom_data_up = bottom[0]->cpu_data();
        const Dtype* bottom_data_down = bottom[1]->cpu_data();
        const Dtype* bottom_data_full = bottom[2]->cpu_data();
        const Dtype* theta_data = bottom[3]->cpu_data();
        //const Dtype* target_data = target_.cpu_data();
        caffe_set(top[0]->count(), (Dtype)0, top_data_up);
        caffe_set(top[1]->count(), (Dtype)0, top_data_down);
        caffe_set(top[2]->count(), (Dtype)0, top_data_full);
        caffe_set(theta_.count(), (Dtype)0, theta_blob_data);
        caffe_set(coordinate_.count(), (Dtype)1, coordinate_blob_data);

        //
        for (int i = 0; i < batch_size; i++)
        {
            Dtype theta_src[3][3];
            memset(theta_src,0,sizeof(Dtype)*9);
            Dtype theta_des[3][3];
            memset(theta_des,0,sizeof(Dtype)*9);
            theta_src[0][0] = theta_data[i * 4];
            theta_src[1][1] = theta_data[i * 4 + 1];
            theta_src[0][2] = theta_data[i * 4 + 2];
            theta_src[1][2] = theta_data[i * 4 + 3];

            //computing inverse of theta
            bool res = GetMatrixInverse(theta_src, 3, theta_des);

            theta_blob_data[0] = theta_des[0][0];
            theta_blob_data[1] = theta_des[1][0];
            theta_blob_data[2] = theta_des[0][1];
            theta_blob_data[3] = theta_des[1][1];
            theta_blob_data[4] = theta_des[0][2];
            theta_blob_data[5] = theta_des[1][2];

            //computing up bbox
            Dtype* coordinates_t_up = top_data_up + i * 4;
            for (int j = 0; j < 2; j ++)
            {
                for (int k = 0; k < 2; k++)
                {
                    coordinate_blob_data[j * 3 + k] = bottom_data_up[j * 2 + k];
                }
            }            
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, 2, 3, (Dtype)1, 
                theta_blob_data, coordinate_blob_data, (Dtype)0, coordinates_t_up);
            //computing down bbox
            Dtype* coordinates_t_down = top_data_down + i * 4;
            for (int j = 0; j < 2; j ++)
            {
                for (int k = 0; k < 2; k++)
                {
                    coordinate_blob_data[j * 3 + k] = bottom_data_down[j * 2 + k];
                }
            }            
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, 2, 3, (Dtype)1, 
                theta_blob_data, coordinate_blob_data, (Dtype)0, coordinates_t_down);
            //computing full bbox
            Dtype* coordinates_t_full = top_data_full + i * 4;
            for (int j = 0; j < 2; j ++)
            {
                for (int k = 0; k < 2; k++)
                {
                    coordinate_blob_data[j * 3 + k] = bottom_data_full[j * 2 + k];
                }
            }            
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 2, 2, 3, (Dtype)1, 
                theta_blob_data, coordinate_blob_data, (Dtype)0, coordinates_t_full);
            //

        }


    }

    template <typename Dtype>
    void SpatialTransformerForwardLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom) {
    }

    //
    template <typename Dtype>
    bool SpatialTransformerForwardLayer<Dtype>::GetMatrixInverse(Dtype src[3][3], int n, Dtype des[3][3])
    {
        Dtype flag=getA(src,n);
        Dtype t[3][3];
        if(flag==0)
        {
            return false;
        }
        else
        {
            getAStart(src,n,t);
            for(int i=0;i<n;i++)
            {
                for(int j=0;j<n;j++)
                {
                    des[i][j]=t[i][j]/flag;
                }

            }
        }


        return true;

    }
    
    //|A|
    template <typename Dtype>
    Dtype SpatialTransformerForwardLayer<Dtype>::getA(Dtype arcs[3][3], int n)
    {
        if(n==1)
        {
            return arcs[0][0];
        }
        Dtype ans = 0;
        Dtype temp[3][3]={0.0};
        int i,j,k;
        for(i=0;i<n;i++)
        {
            for(j=0;j<n-1;j++)
            {
                for(k=0;k<n-1;k++)
                {
                    temp[j][k] = arcs[j+1][(k>=i)?k+1:k];

                }
            }
            Dtype t = getA(temp,n-1);
            if(i%2==0)
            {
                ans += arcs[0][i]*t;
            }
            else
            {
                ans -=  arcs[0][i]*t;
            }
        }
        return ans;
    }
    //A*
    template <typename Dtype>
    void  SpatialTransformerForwardLayer<Dtype>::getAStart(Dtype arcs[3][3],int n,Dtype ans[3][3])
    {
        if(n==1)
        {
            ans[0][0] = 1;
            return;
        }
        int i,j,k,t;
        Dtype temp[3][3];
        for(i=0;i<n;i++)
        {
            for(j=0;j<n;j++)
            {
                for(k=0;k<n-1;k++)
                {
                    for(t=0;t<n-1;t++)
                    {
                        temp[k][t] = arcs[k>=i?k+1:k][t>=j?t+1:t];
                    }
                }


                ans[j][i]  =  getA(temp,n-1);
                if((i+j)%2 == 1)
                {
                    ans[j][i] = - ans[j][i];
                }
            }
        }
    }


#ifdef CPU_ONLY
    STUB_GPU(SpatialTransformerForwardLayer);
#endif

    INSTANTIATE_CLASS(SpatialTransformerForwardLayer);

} // namespace caffe
