#include "env.h"
#include "embedding.h"
#include "vector.h"
#include <cstddef>

#include <sentencepiece_processor.h>


MVec* 
image_to_vector(const int32_t width, 
                      const int32_t height, 
                      const float8 norm_mean_1,
                      const float8 norm_mean_2,
                      const float8 norm_mean_3,
                      const float8 norm_std_1,
                      const float8 norm_std_2,
                      const float8 norm_std_3,
                      const char* image_url)
{
    cv::Mat image;
    cv::Mat image_float;
    MVec*   vec;

    try{
        image = cv::imread(image_url);
    }
    catch(const std::exception& e)
    {
        ereport(ERROR,
					(errmsg(e.what())));
        return NULL;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image_float, CV_32FC3, 1.0/255.0, 0);
    cv::resize(image_float, image_float, cv::Size(width, height));

    auto tensor = torch::from_blob(image_float.data, {1, width, height, 3});
    tensor = tensor.permute({0,3,1,2});
    tensor[0][0] = tensor[0][0].sub_(norm_mean_1).div_(norm_std_1);
    tensor[0][1] = tensor[0][1].sub_(norm_mean_2).div_(norm_std_2);
    tensor[0][2] = tensor[0][2].sub_(norm_mean_3).div_(norm_std_3);

    try{
        vec = tensor_to_vector(tensor);
    }
    catch(const std::exception& e)
    {
        ereport(ERROR,
					(errmsg(e.what())));
        return NULL;
    }
    return vec;
    
}


MVec* 
text_to_vector(const char* piece_model_path, 
               const char* text)
{
    MVec* ret;
    auto max_length = 128;
    torch::DeviceType device_type;

    // load spiece model
    sentencepiece::SentencePieceProcessor process;

    // std::string relative_path = "../../model/spiece.model";
    // std::filesystem::path current_path = std::filesystem::current_path();
    // std::filesystem::path absolute_path = current_path / relative_path;

    //process.LoadOrDie("/home/lhh/postgres-DB4AI/src/udf/src/external_process/../model/spiece.model");
    if(!process.Load(piece_model_path).ok()){
        ereport(ERROR,
					(errmsg("spiece model not exist!")));
    }

    auto opts_data = torch::TensorOptions().dtype(torch::kLong);
    std::vector<int> tis_int_a;
    process.Encode(text, &tis_int_a);

    const std::string cls_token = "[CLS]";
    const std::string sep_token = "[SEP]";
    const std::string pad_token = "<pad>";

    // add abnormal token
    tis_int_a.insert(tis_int_a.begin(), process.PieceToId(cls_token));
    tis_int_a.push_back(process.PieceToId(sep_token));

    // Splice the sentenpiece
    std::vector<long> tis;
    tis.reserve(tis_int_a.size());
    tis.insert(tis.end(), tis_int_a.begin(), tis_int_a.end());
    tis.resize(max_length, process.PieceToId(pad_token));

    // create attention_mask, token_type_ids, and position_ids
    std::vector<long> am(tis_int_a.size(), 1);
    am.resize(max_length, 0);
    std::vector<long> ttis(tis_int_a.size(), 0);
    ttis.resize(max_length, 1);


    torch::Tensor token_ids =
        torch::from_blob(tis.data(), {(long)max_length}, opts_data).clone().to(torch::kFloat32);
    torch::Tensor attention_mask =
        torch::from_blob(am.data(), {(long)max_length}, opts_data).clone().to(torch::kFloat32);
    torch::Tensor token_type_ids =
        torch::from_blob(ttis.data(), {(long)max_length}, opts_data).clone().to(torch::kFloat32);
    torch::Tensor position_ids = torch::arange(0, (long)max_length, opts_data).to(torch::kFloat32);

    //std::vector<torch::IValue> inputs;
    token_ids = torch::unsqueeze(token_ids, 0);
    attention_mask = torch::unsqueeze(attention_mask, 0);
    token_type_ids = torch::unsqueeze(token_type_ids, 0);
    position_ids = torch::unsqueeze(position_ids, 0);

    // img_tensor.push_back(token_ids.to(device_type));
    // img_tensor.push_back(attention_mask.to(device_type));
    // img_tensor.push_back(token_type_ids.to(device_type));
    // img_tensor.push_back(position_ids.to(device_type));

    torch::Tensor stacked_inputs = torch::stack({token_ids, attention_mask, token_type_ids, position_ids}, 1);
    // ereport(INFO,
	// 				(errmsg("stacked_inputs: %d", stacked_inputs.sizes().size())));

    try{
        ret = tensor_to_vector(stacked_inputs);
    }
    catch(const std::exception& e)
    {
        ereport(ERROR,
					(errmsg(e.what())));
        return NULL;
    }
    return ret;
}