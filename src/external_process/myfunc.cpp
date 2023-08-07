#include "myfunc.h"
#include "model_manager.h"

extern "C"{
#include "utils/palloc.h"
#include "utils/builtins.h"
}

#include <cstddef>
#include <sentencepiece_processor.h>

extern ModelManager model_manager;

bool MyProcessImage(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    cv::Mat image;
    cv::Mat image_float;
    char* url = (char*)args[0].ptr;

    image = cv::imread(url);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image_float, CV_32FC3, 1.0/255, 0);
    cv::resize(image_float, image_float, cv::Size(64, 64));

    auto tensor = torch::from_blob(image_float.data, {1, 64, 64, 3});
    tensor = tensor.permute({0,3,1,2});
    tensor[0][0] = tensor[0][0].sub_(0.485).div_(0.229);
    tensor[0][1] = tensor[0][1].sub_(0.456).div_(0.224);
    tensor[0][2] = tensor[0][2].sub_(0.406).div_(0.225);
    
    img_tensor.push_back(tensor);
    return true;
}

bool MyOutPutProcessfloat(torch::jit::IValue& output_tensor, Args* args, float8& result)
{
    auto tensor = output_tensor.toTensor().slice(1, 0, 6);
    
    std::tuple<torch::Tensor,torch::Tensor> res = tensor.sort(1, true);
    torch::Tensor top_scores = std::get<0>(res);

    result = top_scores[0][0].item<float8>();
    return true;
}

bool MyOutPutProcesstext(torch::jit::IValue& output_tensor, Args* args, std::string& result)
{
    std::array<const char*, 6> results{
        "A thick and thin place",
        "Bad selvage",
        "Ball",
        "Broken ends or warp",
        "Hole",
        "Oil spot"
    };

    auto tensor = output_tensor.toTensor().slice(1, 0, 6);
    
    std::tuple<torch::Tensor,torch::Tensor> res = tensor.sort(1, true);
    torch::Tensor top_scores = std::get<1>(res);

    int index = top_scores[0][0].item<int>();    
    result = results[index];

    return true;
}


bool SST2PreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    char* text_a = NULL;
    auto max_length = 128;
    torch::DeviceType device_type;

    if(model_manager.GetModelDeviceType("sst2", device_type)){

    }

    text_a = (char*)args[0].ptr;

    // load spiece model
    sentencepiece::SentencePieceProcessor process;

    // std::string relative_path = "../../model/spiece.model";
    // std::filesystem::path current_path = std::filesystem::current_path();
    // std::filesystem::path absolute_path = current_path / relative_path;

    //process.LoadOrDie("/home/lhh/postgres-DB4AI/src/udf/src/external_process/../model/spiece.model");
    if(!process.Load("/tmp/pgdl/model/spiece.model").ok()){
        return false;
    }

    auto opts_data = torch::TensorOptions().dtype(torch::kLong);
    std::vector<int> tis_int_a;
    process.Encode(text_a, &tis_int_a);

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
        torch::from_blob(tis.data(), {(long)max_length}, opts_data).clone();
    torch::Tensor attention_mask =
        torch::from_blob(am.data(), {(long)max_length}, opts_data).clone();
    torch::Tensor token_type_ids =
        torch::from_blob(ttis.data(), {(long)max_length}, opts_data).clone();
    torch::Tensor position_ids = torch::arange(0, (long)max_length, opts_data);

    //std::vector<torch::IValue> inputs;
    token_ids = torch::unsqueeze(token_ids, 0);
    attention_mask = torch::unsqueeze(attention_mask, 0);
    token_type_ids = torch::unsqueeze(token_type_ids, 0);
    position_ids = torch::unsqueeze(position_ids, 0);

    img_tensor.push_back(token_ids.to(device_type));
    img_tensor.push_back(attention_mask.to(device_type));
    img_tensor.push_back(token_type_ids.to(device_type));
    img_tensor.push_back(position_ids.to(device_type));

    return true;
}

bool SST2OutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    auto tensor = outputs.toTuple()->elements()[0].toTensor();
    result = torch::cat(tensor, 0).argmax(1).item<float8>();
    return true;
}

bool SST2OutputProcessText(torch::jit::IValue& outputs, Args* args, std::string& result)
{
    float8 result_float;
    char*  result_str = NULL;
    auto tensor = outputs.toTuple()->elements()[0].toTensor();
    result_float = torch::cat(tensor, 0).argmax(1).item<float8>();
    if(result_float == 0){
        result = "消极情绪";
    }else if(result_float == 1){
        result = "积极情绪";
    }else{
        return false;
    }
    return true;
}

void register_callback()
{
    elog(INFO, "register callback");
    model_manager.RegisterPreProcess("defect", MyProcessImage);
    model_manager.RegisterOutoutProcessFloat("defect", MyOutPutProcessfloat);
    model_manager.RegisterOutoutProcessText("defect", MyOutPutProcesstext);
    
    model_manager.RegisterPreProcess("sst2", SST2PreProcess);
    model_manager.RegisterOutoutProcessFloat("sst2", SST2OutputProcessFloat);
    model_manager.RegisterOutoutProcessText("sst2", SST2OutputProcessText);
}
