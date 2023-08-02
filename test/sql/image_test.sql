-- 
CREATE TABLE IF NOT EXISTS image_test(
    collection_time TIMESTAMP NOT NULL,
    temperature FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    image_url text NOT NULL);


INSERT INTO image_test(user_name, url)
VALUES
('bob', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_10.jpg'), 
('frank', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_11.jpg'), 
('bob', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_12.jpg'), 
('vicky', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_1.jpg'), 
('frank', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_2.jpg'),
('vicky', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_3.jpg'), 
('jeff', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_4.jpg'), 
('vicky', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_5.jpg'), 
('frank', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_6.jpg'),
('alice', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_7.jpg'),
('alice', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_8.jpg'),
('jeff', '/home/lhh/postgres-DB4AI/src/udf/test/image/img_9.jpg');



CREATE TABLE IF NOT EXISTS defect_classification_results(category integer primary key, category_name text);
INSERT INTO defect_classification_results (category, category_name)
VALUES 
(0, 'A thick and thin place'), 
(1, 'Bad selvage'),
(2, 'Ball'),
(3, 'Broken ends or warp'),
(4, 'Hole'),
(5, 'Oil spot');

SELECT *, pg_predict_float('defect', 'cpu', image_url) 
AS result 
FROM image_test
JOIN defect_classification_results
ON 

SELECT *, pg_predict_text('defect', 'cpu', image_url) 
AS result 
FROM image_test;

