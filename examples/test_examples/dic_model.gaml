model dic

global {
	string st;
	map<string, unknown> dic <- from_json('{"key":123.4}');
	string to_st <- to_json(["key"::123.4]);
	
	init {
//		write "to_st: " + to_st;
		write "dic: " + dic;
	}
}

experiment dic {
	
}