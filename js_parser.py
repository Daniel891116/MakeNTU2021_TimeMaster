def js_parser(string):
	tokens = string.split('_')
	result = {'name':tokens[0],'period':int(tokens[1])}
	return result
