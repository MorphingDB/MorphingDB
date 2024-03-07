create table vec_test(id integer, vec mvec);

insert into vec_test values(1, '[1.0,2.2,3.123,4.2]{4}');
insert into vec_test values(1, ARRAY[1.0,2.0,3.0,1.2345]::float4[]::mvec);

select get_mvec_shape(vec) from vec_test;
select get_mvec_data(vec) from vec_test;

update vec_test set vec=vec+vec;
update vec_test set vec=vec-text_to_mvec('[1,2,3,4]');

select * from vec_test where vec=='[1,2.2,3.123,4.2]';
