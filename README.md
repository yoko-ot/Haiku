## 実行について
	python seq2seq.py -g　で実行
	chainerは1.24.0
	elm15で実行

# dataset

   ------haiku------
	haiku_all
		俳句そのまま(自由律俳句や字余り字足らずの俳句も含まれている)
		62250 data
	haiku_raw
		haiku_allから、17音の俳句だけを抽出
		40000 data
	haiku_raw_test
		haiku_rawのテストデータ。
		haiku_allから17音になるものをとりだすと、40017 dataになった。
		その１７句分をテストデータとして回した。(もっと増やすべき…？)
		17 data
	haiku_kana
		haiku_rawをmecabで仮名にしたもの。
		５－７－５で分かれている。
		40000 data
	haiku_kana_test
		haiku_kanaのテストデータ。
		17 data
	haiku_wakati
		haiku_rawを分かち書きにした。
		40000 data
	haiku_wakati_test
		haiku_wakatiのテストデータ
		17 data



   -------content_word---------
	content
		haiku_rawの内容語を抽出。
		40000 data
	content_test
		contentのテストデータ。
		17 data
	content_kana
		content(内容語)をmecabで仮名にしたもの
		40000 data
	content_kana_test
		content_kanaのテストデータ
		17 data

   -----------kigo-------------
	spring
		春の季語
		1275 data
	summer
		夏の季語
		1628 data
	autumn
		秋の季語
		1033 data
	winter
		冬の季語
		1172 data
	new_year
		新年の季語
223 data
