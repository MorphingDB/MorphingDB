#include "myfunc.h"
#include "model_manager.h"
#include "vector.h"

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
    char* image_url = (char*)args[0].ptr;
    try{
        image = cv::imread(image_url);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image_float, CV_32FC3, 1.0/255.0, 0);
        cv::resize(image_float, image_float, cv::Size(224, 224));

        auto tensor = torch::from_blob(image_float.data, {1, 224, 224, 3});
        tensor = tensor.permute({0,3,1,2});
        tensor[0][0] = tensor[0][0].sub_(0.485).div_(0.225);
        tensor[0][1] = tensor[0][1].sub_(0.485).div_(0.225);
        tensor[0][2] = tensor[0][2].sub_(0.485).div_(0.225);

        auto tensor_copy = tensor.clone();
        img_tensor.emplace_back(tensor_copy);
    }catch(const std::exception& e){
        return false;
    }
    return true;
}

bool MyProcessImage_vec(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    MVec* vector = (MVec*)args[0].ptr;
    torch::Tensor tensor = vector_to_tensor(vector);
    img_tensor.push_back(tensor);

    return true;
}

bool MyOutPutProcessfloat(torch::jit::IValue& output_tensor, Args* args, float8& result)
{
    try{
        auto tensor = output_tensor.toTensor().slice(1, 0, 120);
    
        std::tuple<torch::Tensor,torch::Tensor> res = tensor.sort(1, true);
        torch::Tensor top_scores = std::get<1>(res);

        result = top_scores[0][0].item<float8>();
    } catch(const std::exception& e){
        return false;
    }
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
    if(!process.Load("/home/lhh/pgdl_basemodel_new/model/spiece.model").ok()){
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

bool SST2_VecPreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    MVec* vector = (MVec*)args[0].ptr;
    torch::Tensor tensor = vector_to_tensor(vector);

    std::vector<torch::Tensor> unbound_tensors = tensor.unbind(1);

    torch::Tensor token_ids = unbound_tensors[0].to(torch::kInt64);
    torch::Tensor attention_mask = unbound_tensors[1].to(torch::kInt64);
    torch::Tensor token_type_ids = unbound_tensors[2].to(torch::kInt64);
    torch::Tensor position_ids = unbound_tensors[3].to(torch::kInt64);

    img_tensor.push_back(token_ids);
    img_tensor.push_back(attention_mask);
    img_tensor.push_back(token_type_ids);
    img_tensor.push_back(position_ids);

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

bool IrisPreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    double sepal_length = args[0].floating;
    double sepal_width = args[1].floating;
    double petal_length = args[2].floating;
    double petal_width = args[3].floating;

    auto combined_tensor = torch::tensor({sepal_length, sepal_width, petal_length, petal_width}, torch::kFloat32).reshape({1, 4});
    img_tensor.push_back(combined_tensor);
    return true;
}

bool IrisOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = result_float;
    }else if(result_float == 1){
        result = result_float;
    }else if(result_float == 2){
        result = result_float;
    }else{
        return false;
    }
    return true;
}

bool IrisOutputProcessText(torch::jit::IValue& outputs, Args* args, std::string& result)
{

    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = "Iris-setosa";
    }else if(result_float == 1){
        result = "Iris-versicolor";
    }else if(result_float == 2){
        result = "Iris-virginica";
    }else{
        return false;
    }
    return true;
}

bool SlicePreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    MVec* vector = (MVec*)args[0].ptr;
    torch::Tensor tensor = vector_to_tensor(vector);
    img_tensor.push_back(tensor);

    return true;
}

bool SliceOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    int result_float;
    auto tensor = outputs.toTensor();
    result = tensor.item<float>();
    return true;
}


bool BankMarketPreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    double age = args[0].integer;
    double job = args[1].integer;
    double marital = args[2].integer;
    double education = args[3].integer;
    double default_ = args[4].integer;
    double housing = args[5].integer;
    double loan = args[6].integer;
    double contact = args[7].integer;
    double month = args[8].integer;
    double day_of_week = args[9].integer;
    double duration = args[10].integer;
    double campaign = args[11].integer;
    double pdays = args[12].integer;
    double previous = args[13].integer;
    double poutcome = args[14].integer;
    double emp_var_rate = args[15].floating;
    double cons_price_idx = args[16].floating;
    double cons_conf_idx = args[17].floating;
    double euribor3m = args[18].floating;
    double nr_employed = args[19].floating;

    auto combined_tensor = torch::tensor({age, job, marital, education, default_, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed}, torch::kFloat32).reshape({1, 20});
    img_tensor.push_back(combined_tensor);
    return true;
}

bool BankMarketOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = result_float;
    }else if(result_float == 1){
        result = result_float;
    }else{
        return false;
    }
    return true;
}

bool BankMarketOutputProcessText(torch::jit::IValue& outputs, Args* args, std::string& result)
{

    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = "no";
    }else if(result_float == 1){
        result = "yes";
    }else{
        return false;
    }
    return true;
}

bool CreditPreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    float Index = args[0].floating;
    float id = args[1].floating;
    float customer_id = args[2].floating;
    float month = args[3].floating;
    float age = args[4].floating;
    float ssn = args[5].floating;
    float annual_income = args[6].floating;
    float monthly_inhand_salary = args[7].floating;
    float num_bank_accounts = args[8].floating;
    float num_credit_card = args[9].floating;
    float interest_rate = args[10].floating;
    float num_of_loan = args[11].floating;
    float delay_from_due_date = args[12].floating;
    float num_of_delayed_payment = args[13].floating;
    float changed_credit_limit = args[14].floating;
    float num_credit_inquiries = args[15].floating;
    float outstanding_debt = args[16].floating;
    float credit_utilization_ratio = args[17].floating;
    float credit_history_age = args[18].floating;
    float total_emi_per_month = args[19].floating;
    float amount_invested_monthly = args[0].floating;
    float monthly_balance = args[1].floating;
    float name = args[2].floating;
    float occupation = args[3].floating;
    float credit_mix = args[4].floating;
    float payment_of_min_amount = args[5].floating;
    float payment_behaviour = args[6].floating;
    float auto_loan = args[7].floating;
    float credit_builder_loan = args[8].floating;
    float debt_consolidation_loan = args[9].floating;
    float home_equity_loan  = args[10].floating;
    float mortgage_loan = args[11].floating;
    float not_specified  = args[12].floating;
    float payday_loan = args[13].floating;
    float personal_loan  = args[14].floating;
    float student_loan = args[15].floating;

    auto combined_tensor = torch::tensor({Index,id,customer_id,month,age,ssn,annual_income,monthly_inhand_salary,num_bank_accounts,num_credit_card,interest_rate,num_of_loan,delay_from_due_date,num_of_delayed_payment,changed_credit_limit,num_credit_inquiries,outstanding_debt,credit_utilization_ratio,credit_history_age,total_emi_per_month,amount_invested_monthly,monthly_balance,name,occupation,credit_mix,payment_of_min_amount,payment_behaviour,auto_loan,credit_builder_loan,debt_consolidation_loan,home_equity_loan,mortgage_loan,not_specified,payday_loan,personal_loan,student_loan}, torch::kFloat32).reshape({1, 36});
    img_tensor.push_back(combined_tensor);
    return true;
}

bool CreditOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = result_float;
    }else if(result_float == 1){
        result = result_float;
    }else if(result_float == 2){
        result = result_float;
    }else{
        return false;
    }
    return true;
}

bool CreditOutputProcessText(torch::jit::IValue& outputs, Args* args, std::string& result)
{

    int result_float;
    auto tensor = outputs.toTensor();
    result_float = tensor.argmax(1).item<float8>();

    if(result_float == 0){
        result = "Good";
    }else if(result_float == 1){
        result = "Standard";
    }else if(result_float == 2){
        result = "poor";
    }else{
        return false;
    }
    return true;
}


bool SquardPreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    MVec* vector = (MVec*)args[0].ptr;
    torch::Tensor tensor = vector_to_tensor(vector);

    for (size_t i = 0; i < 4; ++i) {
        torch::Tensor sub_tensor = tensor.unbind(1)[i];
        // 如果需要转换数据类型，使用std::move减少复制
        img_tensor.push_back(sub_tensor);
    }
    return true;
}

bool SquardOutputProcessText(torch::jit::IValue& outputs, Args* args, std::string& result)
{
    // 还需要decode 需要将token_ids的内容拿出来
    try{
        MVec* vector = (MVec*)args[0].ptr;
        torch::Tensor stacked_inputs = vector_to_tensor(vector);
        torch::Tensor token_ids = stacked_inputs[0][0];

        // int* datas = token_ids.data_ptr<int>();

        std::vector<int> tis;
        for(int i=0; i<512; ++i){
            tis.push_back(token_ids[i].item().toInt());
        }

        auto tensor = outputs.toTuple()->elements();

        auto start_index = torch::argmax(tensor[0].toTensor()).item<int>();

        auto end_index = torch::argmax(tensor[1].toTensor()).item<int>();

        sentencepiece::SentencePieceProcessor process;
        process.LoadOrDie("/data/nlp/squad2/albert-base-v2-squad2/spiece.model");
        std::vector<int> answer;
        answer.insert(answer.end(), tis.begin() + start_index, tis.begin() + end_index + 1);
        process.Decode(answer, &result);
    }catch(const std::exception& e){
        return false;
    }
    return true;

}

bool SquardOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    auto tensor = outputs.toTuple()->elements();

    auto start_index = torch::argmax(tensor[0].toTensor()).item<int>();

    auto end_index = torch::argmax(tensor[1].toTensor()).item<int>();

    result = start_index;
    return true;
}

bool FinancePreProcess(std::vector<torch::jit::IValue>& img_tensor, Args* args)
{
    MVec* vector = (MVec*)args[0].ptr;
    torch::Tensor tensor = vector_to_tensor(vector);

    for (size_t i = 0; i < 2; ++i) {
        torch::Tensor sub_tensor = tensor.unbind(1)[i].to(torch::kInt64);
        // 如果需要转换数据类型，使用std::move减少复制
        img_tensor.push_back(sub_tensor);
    }
    return true;
}

bool FinanceOutputProcessFloat(torch::jit::IValue& outputs, Args* args, float8& result)
{
    auto dict = outputs.toGenericDict();
    auto logits = dict.at("logits").toTensor();

    auto index = torch::argmax(logits, 1).item<int>();

    result = index;
    return true;
}

void register_callback()
{
    elog(INFO, "register callback");
    model_manager.RegisterPreProcess("defect", MyProcessImage);
    model_manager.RegisterOutoutProcessFloat("defect", MyOutPutProcessfloat);
    model_manager.RegisterOutoutProcessText("defect", MyOutPutProcesstext);

    model_manager.RegisterPreProcess("defect_vec", MyProcessImage_vec);
    model_manager.RegisterOutoutProcessFloat("defect_vec", MyOutPutProcessfloat);
    model_manager.RegisterOutoutProcessText("defect_vec", MyOutPutProcesstext);

    // model_manager.RegisterPreProcess("defect_1", MyProcessImage);
    // model_manager.RegisterOutoutProcessFloat("defect_1", MyOutPutProcessfloat);
    // model_manager.RegisterOutoutProcessText("defect_1", MyOutPutProcesstext);

    // model_manager.RegisterPreProcess("defect_2", MyProcessImage);
    // model_manager.RegisterOutoutProcessFloat("defect_2", MyOutPutProcessfloat);
    // model_manager.RegisterOutoutProcessText("defect_2", MyOutPutProcesstext);
    
    model_manager.RegisterPreProcess("sst2", SST2PreProcess);
    model_manager.RegisterOutoutProcessFloat("sst2", SST2OutputProcessFloat);
    model_manager.RegisterOutoutProcessText("sst2", SST2OutputProcessText);

    model_manager.RegisterPreProcess("sst2_vec", SST2_VecPreProcess);
    model_manager.RegisterOutoutProcessFloat("sst2_vec", SST2OutputProcessFloat);
    model_manager.RegisterOutoutProcessText("sst2_vec", SST2OutputProcessText);

    model_manager.RegisterPreProcess("iris", IrisPreProcess);
    model_manager.RegisterOutoutProcessFloat("iris", IrisOutputProcessFloat);
    model_manager.RegisterOutoutProcessText("iris", IrisOutputProcessText);

    model_manager.RegisterPreProcess("slice", SlicePreProcess);
    model_manager.RegisterOutoutProcessFloat("slice", SliceOutputProcessFloat);

    model_manager.RegisterPreProcess("year_predict", SlicePreProcess);
    model_manager.RegisterOutoutProcessFloat("year_predict", SliceOutputProcessFloat);

    model_manager.RegisterPreProcess("swarm", SlicePreProcess);
    model_manager.RegisterOutoutProcessFloat("swarm", SliceOutputProcessFloat);

    model_manager.RegisterPreProcess("银行信贷", BankMarketPreProcess);
    model_manager.RegisterOutoutProcessFloat("银行信贷", BankMarketOutputProcessFloat);
    model_manager.RegisterOutoutProcessText("银行信贷", BankMarketOutputProcessText);

    model_manager.RegisterPreProcess("信用卡客户信用评级分类", CreditPreProcess);
    model_manager.RegisterOutoutProcessFloat("信用卡客户信用评级分类", CreditOutputProcessFloat);
    model_manager.RegisterOutoutProcessText("信用卡客户信用评级分类", CreditOutputProcessText);

    model_manager.RegisterPreProcess("googlenet_cifar10", MyProcessImage_vec);
    model_manager.RegisterOutoutProcessFloat("googlenet_cifar10", MyOutPutProcessfloat);

    model_manager.RegisterPreProcess("alexnet_stanford_dogs", MyProcessImage_vec);
    model_manager.RegisterOutoutProcessFloat("alexnet_stanford_dogs", MyOutPutProcessfloat);
    // model_manager.RegisterOutoutProcessText("googlenet_cifar10", CreditOutputProcessText);

    model_manager.RegisterPreProcess("squard", SquardPreProcess);
    model_manager.RegisterOutoutProcessFloat("squard", SquardOutputProcessFloat);
    model_manager.RegisterOutoutProcessText("squard", SquardOutputProcessText);

    model_manager.RegisterPreProcess("finance", FinancePreProcess);
    model_manager.RegisterOutoutProcessFloat("finance", FinanceOutputProcessFloat);
}
