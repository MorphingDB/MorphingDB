-- create model
SELECT create_model('sst2', '/home/lhh/postgres-DB4AI/src/udf/model/traced_albert.pt', 'nlp classification test');
SELECT create_model('defect', '/home/lhh/postgres-DB4AI/src/udf/model/model.pt', 'image classification test');

-- create nlp text table
CREATE TABLE IF NOT EXISTS nlp_test(user_name text, comment text);
-- add comment
INSERT INTO nlp_test (user_name, comment)
VALUES
('bob', 'it takes a strange kind of laziness to waste the talents of robert forster , anne meara , eugene levy , and reginald veljohnson all in the same movie . '),
('alice', '... the film suffers from a lack of humor ( something needed to balance out the violence ) ... '),
('bob', 'a gorgeous , high-spirited musical from india that exquisitely blends music , dance , song , and high drama . '),
('jeff', 'pumpkin takes an admirable look at the hypocrisy of political correctness , but it does so with such an uneven tone that you never know when humor ends and tragedy begins . '),
('frank', 'the iditarod lasts for days - this just felt like it did . '),
('alice', 'a delectable and intriguing thriller filled with surprises , read my lips is an original . '),
('jeff', 'seldom has a movie so closely matched the spirit of a man and his work . '),
('frank', 'nicks , seemingly uncertain what ''s going to make people laugh , runs the gamut from stale parody to raunchy sex gags to formula romantic comedy . '),
('albert', 'the action switches between past and present , but the material link is too tenuous to anchor the emotional connections that purport to span a 125-year divide . '),
('frank', 'it ''s an offbeat treat that pokes fun at the democratic exercise while also examining its significance for those who take part . '),
('jeff', 'it ''s a cookie-cutter movie , a cut-and-paste job . '),
('alice', 'thanks to scott ''s charismatic roger and eisenberg ''s sweet nephew , roger dodger is one of the most compelling variations on in the company of men . '),
('vicky', '... designed to provide a mix of smiles and tears , `` crossroads '' instead provokes a handful of unintentional howlers and numerous yawns . ');


-- create classification_results table
CREATE TABLE IF NOT EXISTS classification_results(category integer primary key, category_name text);

-- add category
INSERT INTO classification_results (category, category_name)
VALUES (0, '消极情绪'), (1, '积极情绪');

-- register callback
SELECT register_process();


-- extension test result
SELECT comment, classification_results.category_name  
FROM nlp_test 
JOIN classification_results 
ON predict_float('sst2', 'cpu', comment) = classification_results.category;

SELECT comment,predict_text('sst2', 'cpu', comment) 
AS result 
FROM nlp_test;

SELECT image_test.user_name,predict_text('defect', 'cpu', url) 
AS result 
FROM image_test;

SELECT nlp_test.user_name, nlp_test.comment
FROM nlp_test
JOIN image_test
ON image_test.user_name = nlp_test.user_name
WHERE predict_text('sst2', 'cpu', nlp_test.comment)='消极情绪'
AND predict_text('defect', 'cpu', image_test.url)='Hole';



-- kernel

-- create model
CREATE MODEL sst2 
PATH '/home/lhh/model/traced_albert.pt' 
DESCRIPTION 'nlp test';

CREATE MODEL defect 
PATH '/home/lhh/model/model.pt' 
DESCRIPTION 'image classification test';

-- create udf

CREATE OR REPLACE FUNCTION register_process(OUT void)
    AS '/home/lhh/pg_kernel/src/udf/build/pgdl.so', 'RegisterCallback' LANGUAGE C STRICT;


-- 注册输入输出处理函数
SELECT
register_process();


SELECT comment, classification_results.category_name  
FROM nlp_test 
JOIN classification_results 
ON pg_predict_float('sst2', 'cpu', comment) = classification_results.category;

SELECT comment, pg_predict_text('sst2', 'cpu', comment) 
AS result 
FROM nlp_test;

SELECT image_test.user_name, pg_predict_text('defect', 'cpu', url) 
AS result 
FROM image_test;


SELECT nlp_test.user_name, nlp_test.comment
FROM nlp_test
JOIN image_test
ON image_test.user_name = nlp_test.user_name
WHERE pg_predict_text('sst2', 'cpu', nlp_test.comment)='消极情绪'
AND pg_predict_text('defect', 'cpu', image_test.url)='Hole';