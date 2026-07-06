scores = [
"ruler_niah__65536::suite: 0.514687",
    "ruler_multi_hop_tracing__65536::suite: 0.312",
    "ruler_aggregation__65536::suite: 0.0771667",
    "ruler_qa__65536::suite: 0.23",
    "ruler_all__65536::suite: 0.387987",
    "ruler_cwe__65536::std: 0.011",
    "ruler_fwe__65536::std: 0.143333",
    "ruler_niah_mk_1__65536::std: 0.43",
    "ruler_niah_mk_2__65536::std: 0.11",
    "ruler_niah_mk_3__65536::std: 0.07",
    "ruler_niah_mq__65536::std: 0.435",
    "ruler_niah_mv__65536::std: 0.4825",
    "ruler_niah_s_1__65536::std: 1.0",
    "ruler_niah_s_2__65536::std: 0.76",
    "ruler_niah_s_3__65536::std: 0.83",
    "ruler_qa_1__65536::std: 0.24",
    "ruler_qa_2__65536::std: 0.22",
    "ruler_vt__65536::std: 0.312",
    "ruler_niah__32768::suite: 0.604687",
    "ruler_multi_hop_tracing__32768::suite: 0.068",
    "ruler_aggregation__32768::suite: 0.163167",
    "ruler_qa__32768::suite: 0.29",
    "ruler_all__32768::suite: 0.447064",
    "ruler_cwe__32768::std: 0.033",
    "ruler_fwe__32768::std: 0.293333",
    "ruler_niah_mk_1__32768::std: 0.56",
    "ruler_niah_mk_2__32768::std: 0.28",
    "ruler_niah_mk_3__32768::std: 0.09",
    "ruler_niah_mq__32768::std: 0.6575",
    "ruler_niah_mv__32768::std: 0.63",
    "ruler_niah_s_1__32768::std: 1.0",
    "ruler_niah_s_2__32768::std: 0.9",
    "ruler_niah_s_3__32768::std: 0.72",
    "ruler_qa_1__32768::std: 0.31",
    "ruler_qa_2__32768::std: 0.27",
    "ruler_vt__32768::std: 0.068",
    "ruler_niah__16384::suite: 0.740312",
    "ruler_multi_hop_tracing__16384::suite: 0.17",
    "ruler_aggregation__16384::suite: 0.178333",
    "ruler_qa__16384::suite: 0.335",
    "ruler_all__16384::suite: 0.547628",
    "ruler_cwe__16384::std: 0.03",
    "ruler_fwe__16384::std: 0.326667",
    "ruler_niah_mk_1__16384::std: 0.73",
    "ruler_niah_mk_2__16384::std: 0.58",
    "ruler_niah_mk_3__16384::std: 0.16",
    "ruler_niah_mq__16384::std: 0.8525",
    "ruler_niah_mv__16384::std: 0.8",
    "ruler_niah_s_1__16384::std: 1.0",
    "ruler_niah_s_2__16384::std: 1.0",
    "ruler_niah_s_3__16384::std: 0.8",
    "ruler_qa_1__16384::std: 0.4",
    "ruler_qa_2__16384::std: 0.27",
    "ruler_vt__16384::std: 0.17",
    "ruler_niah__8192::suite: 0.859063",
    "ruler_multi_hop_tracing__8192::suite: 0.254",
    "ruler_aggregation__8192::suite: 0.241333",
    "ruler_qa__8192::suite: 0.5",
    "ruler_all__8192::suite: 0.662244",
    "ruler_cwe__8192::std: 0.046",
    "ruler_fwe__8192::std: 0.436667",
    "ruler_niah_mk_1__8192::std: 0.81",
    "ruler_niah_mk_2__8192::std: 0.84",
    "ruler_niah_mk_3__8192::std: 0.48",
    "ruler_niah_mq__8192::std: 0.8975",
    "ruler_niah_mv__8192::std: 0.905",
    "ruler_niah_s_1__8192::std: 1.0",
    "ruler_niah_s_2__8192::std: 1.0",
    "ruler_niah_s_3__8192::std: 0.94",
    "ruler_qa_1__8192::std: 0.59",
    "ruler_qa_2__8192::std: 0.41",
    "ruler_vt__8192::std: 0.254",
     "ruler_niah__4096::suite: 0.956875",
    "ruler_multi_hop_tracing__4096::suite: 0.874",
    "ruler_aggregation__4096::suite: 0.5145",
    "ruler_qa__4096::suite: 0.535",
    "ruler_all__4096::suite: 0.817538",
    "ruler_cwe__4096::std: 0.359",
    "ruler_fwe__4096::std: 0.67",
    "ruler_niah_mk_1__4096::std: 0.95",
    "ruler_niah_mk_2__4096::std: 0.96",
    "ruler_niah_mk_3__4096::std: 0.78",
    "ruler_niah_mq__4096::std: 0.98",
    "ruler_niah_mv__4096::std: 0.985",
    "ruler_niah_s_1__4096::std: 1.0",
    "ruler_niah_s_2__4096::std: 1.0",
    "ruler_niah_s_3__4096::std: 1.0",
    "ruler_qa_1__4096::std: 0.72",
    "ruler_qa_2__4096::std: 0.35",
    "ruler_vt__4096::std: 0.874"
    ]
prefix = "ruler_"
lens = [4096, 8192, 16384, 32768, 65536]
tasks = ["niah_s_1", "niah_s_2", "niah_s_3", \
         "niah_mk_1", "niah_mk_2", "niah_mk_3", "niah_mv", \
            "niah_mq", "vt", "cwe", "fwe", "qa_1", "qa_2"]

def extract_val(resultstr):
    #if "4096" not in resultstr and "32768" not in resultstr:
    #    return ""
    this_str = [s for s in scores if s.startswith(resultstr)]
    assert len(this_str) == 1, f"missing {resultstr}"
    this_str = this_str[0]
    this_str = str(float(this_str.split(":")[-1].strip())*100)
    return this_str

result = ""
for length in lens:
    for task in tasks:
        result += extract_val(f"{prefix}{task}__{length}") + ","
    result += "\n"

print(result)