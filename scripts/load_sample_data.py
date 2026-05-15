"""Generate and load a small synthetic dataset for development.

This produces one XML file with three PTMs that imitate the SMR-Pro
structure: a "warm floor" kit, a "wall painting" kit, and a
"roof pie" kit. Good enough to exercise the full pipeline end-to-end.
"""
from __future__ import annotations

from pathlib import Path

from common.config import RAW_DIR
from common.db import init_schema
from common.logging_config import get_logger
from training.ingest_xml import ingest

log = get_logger(__name__)

SAMPLE_XML = """<?xml version="1.0" encoding="utf-8"?>
<stroyka>
  <ob_smeta glava="Глава 6. Внутренние работы">
    <loc_smeta>

      <ptm kod="Ж1-01" naim="Тёплый пол в жилом помещении">
        <rascenka obosn="ГЭСН07-05-011" naim="Укладка нагревательного кабеля" tip="100" ed_izm="м"/>
        <rascenka obosn="ГЭСН07-03-002" naim="Устройство теплоизоляционного слоя из плит" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН07-02-011" naim="Устройство цементно-песчаной стяжки толщиной 50 мм" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН07-01-005" naim="Огрунтовка поверхности бетоноконтактом" tip="100" ed_izm="м2"/>
        <rascenka obosn="М-001" naim="Кабель нагревательный одножильный" tip="101" ed_izm="м"/>
      </ptm>

      <ptm kod="Ж1-02" naim="Тёплый пол в санузле">
        <rascenka obosn="ГЭСН07-05-011" naim="Укладка нагревательного кабеля" tip="100" ed_izm="м"/>
        <rascenka obosn="ГЭСН07-03-002" naim="Устройство теплоизоляционного слоя из плит" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН07-02-011" naim="Устройство цементно-песчаной стяжки толщиной 50 мм" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН07-04-001" naim="Гидроизоляция обмазочная битумной мастикой" tip="100" ed_izm="м2"/>
      </ptm>

      <ptm kod="Ж2-10" naim="Окраска стен поливинилацетатными составами">
        <rascenka obosn="ГЭСН15-04-005" naim="Окраска поливинилацетатными водоэмульсионными составами улучшенная по штукатурке стен" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН15-02-016" naim="Шпатлёвка финишная по штукатурке" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН15-01-019" naim="Огрунтовка поверхности под окраску" tip="100" ed_izm="м2"/>
        <rascenka obosn="М-010" naim="Краска водоэмульсионная белая" tip="101" ed_izm="кг"/>
      </ptm>

      <ptm kod="Ж2-11" naim="Окраска стен в кухне">
        <rascenka obosn="ГЭСН15-04-005" naim="Окраска поливинилацетатными водоэмульсионными составами улучшенная по штукатурке стен" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН15-02-016" naim="Шпатлёвка финишная по штукатурке" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН15-01-019" naim="Огрунтовка поверхности под окраску" tip="100" ed_izm="м2"/>
      </ptm>

      <ptm kod="К3-05" naim="Кровельный пирог скатной кровли">
        <rascenka obosn="ГЭСН12-01-001" naim="Устройство пароизоляции из плёнки полиэтиленовой" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-003" naim="Утепление кровли минераловатными плитами" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-005" naim="Устройство гидроветрозащитной мембраны" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-011" naim="Устройство обрешётки из бруса хвойных пород" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-020" naim="Покрытие кровли металлочерепицей" tip="100" ed_izm="м2"/>
        <rascenka obosn="М-020" naim="Плита минераловатная теплоизоляционная" tip="101" ed_izm="м3"/>
      </ptm>

      <ptm kod="К3-06" naim="Кровельный пирог мансарды">
        <rascenka obosn="ГЭСН12-01-001" naim="Устройство пароизоляции из плёнки полиэтиленовой" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-003" naim="Утепление кровли минераловатными плитами" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-005" naim="Устройство гидроветрозащитной мембраны" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН12-01-020" naim="Покрытие кровли металлочерепицей" tip="100" ed_izm="м2"/>
      </ptm>

      <ptm kod="Ш5-01" naim="Штукатурка стен по маякам">
        <rascenka obosn="ГЭСН15-02-001" naim="Штукатурка цементно-известковая по маякам" tip="100" ed_izm="м2"/>
        <rascenka obosn="ГЭСН15-01-019" naim="Огрунтовка поверхности под окраску" tip="100" ed_izm="м2"/>
        <rascenka obosn="М-030" naim="Смесь сухая штукатурная цементная" tip="101" ed_izm="т"/>
      </ptm>

    </loc_smeta>
  </ob_smeta>
</stroyka>
"""


def main() -> None:
    init_schema()

    sample_path = RAW_DIR / "sample.xml"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(SAMPLE_XML, encoding="utf-8")
    log.info("Wrote sample XML to %s", sample_path)

    rascenki_count, tx_count = ingest(RAW_DIR)
    log.info("Loaded %d rascenki, %d transactions", rascenki_count, tx_count)


if __name__ == "__main__":
    main()
