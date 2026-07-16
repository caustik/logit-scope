function(logit_scope_hex_file input output_variable)
    file(READ "${input}" file_hex HEX)
    string(REGEX REPLACE "(..)" "0x\\1," file_hex "${file_hex}")
    set(${output_variable} "${file_hex}" PARENT_SCOPE)
endfunction()

logit_scope_hex_file("${LOGIT_SCOPE_WEB_DIR}/index.html" LOGIT_SCOPE_INDEX_HTML)
logit_scope_hex_file("${LOGIT_SCOPE_WEB_DIR}/app.js" LOGIT_SCOPE_APP_JS)
logit_scope_hex_file("${LOGIT_SCOPE_WEB_DIR}/style.css" LOGIT_SCOPE_STYLE_CSS)

configure_file("${LOGIT_SCOPE_TEMPLATE}" "${LOGIT_SCOPE_OUTPUT}" @ONLY)
