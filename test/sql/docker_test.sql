-- create model
SELECT create_model('sst2', '/home/pgdl/model/traced_albert.pt', 'nlp classification test');
SELECT create_model('defect', '/home/pgdl/model/model.pt', 'image classification test');

-- create nlp test table
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

-- create image test table
CREATE TABLE IF NOT EXISTS image_test(
    user_name text NOT NULL,
    image_url text NOT NULL);

-- add image
INSERT INTO image_test(user_name, image_url)
VALUES
('bob', '/home/pgdl/test/image/img_10.jpg'), 
('frank', '/home/pgdl/test/image/img_11.jpg'), 
('bob', '/home/pgdl/test/image/img_12.jpg'), 
('vicky', '/home/pgdl/test/image/img_1.jpg'), 
('frank', '/home/pgdl/test/image/img_2.jpg'),
('vicky', '/home/pgdl/test/image/img_3.jpg'), 
('jeff', '/home/pgdl/test/image/img_4.jpg'), 
('vicky', '/home/pgdl/test/image/img_5.jpg'), 
('frank', '/home/pgdl/test/image/img_6.jpg'),
('alice', '/home/pgdl/test/image/img_7.jpg'),
('alice', '/home/pgdl/test/image/img_8.jpg'),
('jeff', '/home/pgdl/test/image/img_9.jpg');


-- create image classification table
CREATE TABLE IF NOT EXISTS defect_classification_results(category integer primary key, category_name text);
INSERT INTO defect_classification_results (category, category_name)
VALUES 
(0, 'A thick and thin place'), 
(1, 'Bad selvage'),
(2, 'Ball'),
(3, 'Broken ends or warp'),
(4, 'Hole'),
(5, 'Oil spot');


-- register callback
SELECT register_process();


-- test
SELECT comment, classification_results.category_name  
FROM nlp_test 
JOIN classification_results 
ON predict_float('sst2', 'cpu', comment) = classification_results.category;

SELECT comment,predict_text('sst2', 'cpu', comment) 
AS result 
FROM nlp_test;

SELECT *, defect_classification_results.category_name
AS result 
FROM image_test
JOIN defect_classification_results
ON predict_text('defect', 'cpu', image_url) = defect_classification_results.category_name;

SELECT *, predict_text('defect', 'cpu', image_url) 
AS result 
FROM image_test;


-- batch test
SELECT nlp_test_2.comment, classification_results.category_name  
FROM (select user_name, comment, predict_batch_float8('sst2', 'cpu', comment) over (rows between current row and 15 following) as comment_2 from nlp_test ) as nlp_test_2
JOIN classification_results
ON nlp_test_2.comment_2 = classification_results.category;

SELECT comment,predict_batch_text('sst2', 'cpu', comment) over (rows between current row and 15 following)
AS result 
FROM nlp_test;

SELECT *, defect_classification_results.category_name
AS result 
FROM (select predict_batch_text('defect', 'cpu', image_url) over (rows between current row and 15 following) from image_test) as image_test_2(pred_cat)
JOIN defect_classification_results
ON pred_cat = defect_classification_results.category_name;

SELECT *, predict_batch_text('defect', 'cpu', image_url) over (rows between current row and 15 following)
AS result 
FROM image_test;

